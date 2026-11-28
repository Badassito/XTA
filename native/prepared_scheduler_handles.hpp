// Owned Linux file descriptions for prepared detector packets. Socket I/O runs
// only in the native actor writer. Nothing here calls into the Python runtime.
#pragma once
#include <array>
#include <charconv>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#ifdef __linux__
#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace xta_scheduler {
constexpr size_t detector_chunk_size = 253;  // Linux SCM_MAX_FD per message
struct DetectorFile {
    int fd = -1;
    ~DetectorFile() {
#ifdef __linux__
        if (fd >= 0) close(fd);
#endif
    }
};
struct DetectorHandle {
    std::shared_ptr<DetectorFile> file;
    std::string identity;
    bool persistent;
};
using DetectorHandles = std::vector<DetectorHandle>;
using OutgoingFiles = std::vector<std::pair<std::uint32_t, std::shared_ptr<DetectorFile>>>;

class DetectorFilePool {
    using FileIdentity = std::array<std::uintmax_t, 6>;

    static bool matches(std::string_view identity, const FileIdentity &fields) {
        for (size_t i = 0; i < fields.size(); ++i) {
            const auto delimiter = identity.find(':');
            const bool last = i + 1 == fields.size();
            if (last != (delimiter == std::string_view::npos)) return false;
            const auto part = identity.substr(0, delimiter);
            if (part.empty() || (part.size() > 1 && part.front() == '0')) return false;
            std::uintmax_t value;
            const auto end = part.data() + part.size();
            const auto result = std::from_chars(part.data(), end, value);
            if (result.ec != std::errc{} || result.ptr != end || value != fields[i]) return false;
            if (!last) identity.remove_prefix(delimiter + 1);
        }
        return true;
    }

    std::mutex mutex_;
    std::map<std::pair<FileIdentity, bool>, std::weak_ptr<DetectorFile>> files_;
    size_t added_ = 0;
public:
    DetectorHandle pin(int source, const std::string &identity, int role) {
#ifdef __linux__
        struct stat info{};
        if (role < 0 || role > 2 || fstat(source, &info) != 0 || !S_ISREG(info.st_mode))
            throw std::runtime_error("invalid detector file registration");
        const FileIdentity proof{
            static_cast<std::uintmax_t>(info.st_dev), static_cast<std::uintmax_t>(info.st_ino),
            static_cast<std::uintmax_t>(info.st_size), static_cast<std::uintmax_t>(info.st_mode),
            static_cast<std::uintmax_t>(info.st_uid), static_cast<std::uintmax_t>(info.st_gid)
        };
        if (!matches(identity, proof)) throw std::runtime_error("detector file identity changed during preparation");
        bool writable = role == 2;
        int mode = fcntl(source, F_GETFL);
        if (mode < 0 || (mode & O_ACCMODE) == O_WRONLY || (writable && (mode & O_ACCMODE) != O_RDWR))
            throw std::runtime_error("detector owner lacks the requested file access");
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = std::make_pair(proof, writable);
        if (auto existing = files_[key].lock()) return {existing, identity, role == 0};
        auto file = std::make_shared<DetectorFile>();
        if ((mode & O_ACCMODE) == (writable ? O_RDWR : O_RDONLY))
            file->fd = fcntl(source, F_DUPFD_CLOEXEC, 0);
        else {
            auto path = "/proc/self/fd/" + std::to_string(source);
            file->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        }
        if (file->fd < 0) throw std::runtime_error("could not pin detector file description");
        files_[key] = file;
        if (++added_ % 128 == 0) {
            for (auto it = files_.begin(); it != files_.end();) {
                if (it->second.expired()) it = files_.erase(it);
                else ++it;
            }
        }
        return {file, identity, role == 0};
#else
        throw std::runtime_error("detector file descriptor transport requires Linux");
#endif
    }
};

#ifdef __linux__
inline size_t send_detector_chunk(int socket, char *slots, size_t count, const OutgoingFiles &files, size_t first) {
    alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(detector_chunk_size * sizeof(int))> control{};
    iovec iov{slots, count * 4};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.data();
    msg.msg_controllen = CMSG_SPACE(count * sizeof(int));
    auto *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(count * sizeof(int));
    auto *fds = reinterpret_cast<int *>(CMSG_DATA(cmsg));
    for (size_t i = 0; i < count; ++i) {
        const auto &entry = files[first + i];
        fds[i] = entry.second->fd;
        for (unsigned j = 0; j < 4; ++j)
            slots[i * 4 + j] = static_cast<char>((entry.first >> (8 * j)) & 255);
    }
    ssize_t n;
    do { n = sendmsg(socket, &msg, MSG_NOSIGNAL); } while (n < 0 && errno == EINTR);
    if (n <= 0) throw std::runtime_error("detector descriptor send failed");
    return static_cast<size_t>(n);
}
#endif
} // namespace xta_scheduler
