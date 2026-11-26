// Native ownership of prepared TTA work, compute windows and retained storage.
#pragma once
#include "prepared_scheduler_handles.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#else
#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace xta_scheduler {
using Id = std::int64_t;
} // namespace xta_scheduler
#include "device_stage.hpp"
namespace xta_scheduler {
using Bytes = std::string;
using Lock = std::unique_lock<std::mutex>;
constexpr Id wire_magic = 0x315343415458LL; // XTACS1
enum Message : Id { dispatch = 1, control, installed, compute, final, failed, control_reply };
constexpr Id max_payload = 64 * 1024 * 1024;
inline void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket invalid_socket = INVALID_SOCKET;
inline void interrupt_socket(Socket socket) { shutdown(socket, SD_BOTH); }
inline void close_socket(Socket socket) { closesocket(socket); }
class SocketEvent {
    WSAEVENT event_ = WSACreateEvent();
public:
    SocketEvent() { require(event_ != WSA_INVALID_EVENT, "failed to create scheduler socket event"); }
    SocketEvent(const SocketEvent &) = delete;
    ~SocketEvent() { WSACloseEvent(event_); }
    WSAEVENT get() const { return event_; }
    void signal() { WSASetEvent(event_); }
};
inline int transfer_chunk(Socket socket, char *data, int size, bool writing, const SocketEvent &stop) {
    // A shutdown(SD_BOTH) on a duplicated Windows socket can leave its blocking
    // recv pending until the peer closes. Own a cancellable overlapped operation.
    thread_local SocketEvent completed;
    WSAResetEvent(completed.get());
    WSAOVERLAPPED operation{};
    operation.hEvent = completed.get();
    WSABUF buffer{static_cast<ULONG>(size), data};
    DWORD done = 0, flags = 0;
    int status = writing ? WSASend(socket, &buffer, 1, &done, 0, &operation, nullptr)
                         : WSARecv(socket, &buffer, 1, &done, &flags, &operation, nullptr);
    if (status == SOCKET_ERROR && WSAGetLastError() == WSA_IO_PENDING) {
        const WSAEVENT events[]{stop.get(), completed.get()};
        auto fired = WSAWaitForMultipleEvents(2, events, FALSE, WSA_INFINITE, FALSE);
        if (fired != WSA_WAIT_EVENT_0 + 1) {
            CancelIoEx(reinterpret_cast<HANDLE>(socket), &operation);
            // The kernel can keep using the buffer and OVERLAPPED until completion.
            WSAGetOverlappedResult(socket, &operation, &done, TRUE, &flags);
            throw std::runtime_error("worker IO interrupted");
        }
        status = WSAGetOverlappedResult(socket, &operation, &done, FALSE, &flags) ? 0 : SOCKET_ERROR;
    }
    return status == SOCKET_ERROR ? -1 : static_cast<int>(done);
}
#else
using Socket = int;
constexpr Socket invalid_socket = -1;
inline void interrupt_socket(Socket socket) { shutdown(socket, SHUT_RDWR); }
inline void close_socket(Socket socket) { close(socket); }
struct SocketEvent { void signal() {} };
#endif

inline void transfer(Socket socket, char *data, size_t size, bool writing, const SocketEvent &stop) {
    while (size) {
        int chunk = static_cast<int>(std::min<size_t>(size, 1 << 20));
#ifdef _WIN32
        int done = transfer_chunk(socket, data, chunk, writing, stop);
#else
        auto done = writing ? send(socket, data, chunk, MSG_NOSIGNAL) : recv(socket, data, chunk, 0);
        if (done < 0 && errno == EINTR) continue;
#endif
        if (done <= 0) throw std::runtime_error(writing ? "worker command send failed" : "worker receipt stream closed");
        data += done;
        size -= static_cast<size_t>(done);
    }
}

struct Lease { Id run, parent, task, worker, epoch, pid, generation; };
struct Packet {
    Id kind;
    Lease lease;
    Bytes data;
    OutgoingFiles files;
};
using Header = std::array<unsigned char, 80>;
inline void encode(Header &buffer, size_t index, Id value) {
    for (size_t i = 0; i < 8; ++i) {
        buffer[index * 8 + i] = static_cast<unsigned char>(static_cast<std::uint64_t>(value) >> (8 * i));
    }
}
inline Id decode(const Header &buffer, size_t index) {
    std::uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) value |= std::uint64_t(buffer[index * 8 + i]) << (8 * i);
    require(value <= static_cast<std::uint64_t>(std::numeric_limits<Id>::max()), "invalid unsigned wire field");
    return static_cast<Id>(value);
}
inline void send_packet(Socket socket, Packet &packet, const SocketEvent &stop) {
    const auto &l = packet.lease;
    Id wire_kind = packet.kind | (static_cast<Id>(packet.files.size()) << 8);
    std::array<Id, 10> fields{wire_magic, wire_kind, l.run, l.parent, l.task,
                              l.worker, l.epoch, l.pid, l.generation, static_cast<Id>(packet.data.size())};
    Header header{};
    for (size_t i = 0; i < fields.size(); ++i) encode(header, i, fields[i]);
    transfer(socket, reinterpret_cast<char *>(header.data()), header.size(), true, stop);
#ifdef __linux__
    for (size_t first = 0; first < packet.files.size(); first += detector_chunk_size) {
        auto count = std::min(detector_chunk_size, packet.files.size() - first);
        std::array<char, detector_chunk_size * 4> slots{};
        auto sent = send_detector_chunk(socket, slots.data(), count, packet.files, first);
        transfer(socket, slots.data() + sent, 4 * count - sent, true, stop);
    }
#endif
    transfer(socket, packet.data.data(), packet.data.size(), true, stop);
}
inline std::pair<Packet, Id> read_header(Socket socket, const SocketEvent &stop) {
    Header header{};
    transfer(socket, reinterpret_cast<char *>(header.data()), header.size(), false, stop);
    require(decode(header, 0) == wire_magic, "invalid scheduler wire protocol");
    Packet packet{decode(header, 1),
                  {decode(header, 2), decode(header, 3), decode(header, 4), decode(header, 5),
                   decode(header, 6), decode(header, 7), decode(header, 8)}, {}};
    Id size = decode(header, 9);
    require(size <= max_payload, "receipt exceeds scheduler payload bound");
    return {std::move(packet), size};
}

using Decision = std::tuple<Id, Id, Id>; // priority, insertion order, identity
struct Parent {
    Id id, expected, bytes, priority, exclusive_kind, headroom;
    bool allow_oversize_host;
    Id pressure_bytes;
    bool descriptor_only;
    Id admission_reserve;
    Id generation = 0, worker = -1, submitted = 0, completed = 0, task_bytes = 0, unread_finals = 0;
    bool funded = false, preparing = false, published = false, retired = false, postprocess_adopted = false;
    std::set<Decision> planned_tasks;
};
struct Task {
    Id id, parent, kinds, priority, required_worker, bytes;
    Bytes data;
    Id ready_since_ns = 0;
    Id generation = 0, worker = -1;
    bool acknowledged = false, compute_seen = false, window_returned = false;
    bool final_seen = false, retired = false;
    DetectorHandles handles;
};
struct Worker {
    Id id, epoch, pid, kinds, window, device;
    Socket socket;
    Id active = 0, unsettled = 0, resident = -1, last_use = 0;
    bool closing = false;
    std::deque<Packet> commands;
    std::set<Id> outstanding_controls;
    std::set<std::string> installed_sources, pending_sources;
    std::thread reader, writer;
    Worker(Id number, Id incarnation, Id process, Id capabilities, Id slots, Id accelerator, Socket channel)
        : id(number), epoch(incarnation), pid(process), kinds(capabilities),
          window(slots), device(accelerator), socket(channel) {}
    ~Worker() { if (socket != invalid_socket) close_socket(socket); }
};
using Summary = std::map<std::string, Id>;

class Scheduler {
    SocketEvent stop_io_;
    DetectorFilePool detector_files_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    const Id run_;
    Id capacity_, tickets_, inference_capacity_, preparation_window_, pressure_capacity_;
    Id used_ = 0, funded_ = 0, inf_bytes_ = 0, inf_parents_ = 0, retained_parent_bytes_ = 0;
    Id sequence_ = 0, generations_ = 0, issued_ = 0, installed_ = 0, finals_ = 0;
    bool dirty_ = false, stopping_ = false;
    std::string fault_;
    std::map<Id, std::unique_ptr<Worker>> workers_;
    std::map<Id, Parent> parents_;
    std::map<Id, Task> tasks_;
    std::set<Decision> pending_parents_, ready_;
    std::deque<std::pair<Id, Id>> preparation_, publication_;
    std::deque<Packet> events_;
    DeviceStages stages_;
    bool last_backlog_ = false, last_pressure_ = false;
    Id queue_ns_ = 0, receipt_refill_ns_ = 0, receipt_refills_ = 0, last_receipt_ns_ = 0;
    std::thread dispatcher_;
    static Id now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void healthy() const {
        if (!fault_.empty()) throw std::runtime_error(fault_);
        require(!stopping_, "native scheduler is closed");
    }
    void wake() {
        dirty_ = true;
        changed_.notify_all();
    }
    void fail(const std::string &reason) {
        if (fault_.empty()) {
            fault_ = reason;
            stop_io_.signal();
            discard_detector_files();
            for (auto &entry : workers_) interrupt_socket(entry.second->socket);
        }
        changed_.notify_all();
    }
    void discard_detector_files() {
        for (auto &entry : tasks_) entry.second.handles.clear();
        for (auto &entry : workers_)
            for (auto &command : entry.second->commands) command.files.clear();
    }
    Id next_generation() {
        require(generations_ < std::numeric_limits<Id>::max(), "scheduler lease generations exhausted");
        return ++generations_;
    }
    Parent &parent(Id id) {
        auto it = parents_.find(id);
        require(it != parents_.end(), "unknown scheduler parent");
        return it->second;
    }
    Task &task(Id id) {
        auto it = tasks_.find(id);
        require(it != tasks_.end(), "unknown scheduler task");
        return it->second;
    }
    Worker &worker(Id id) {
        auto it = workers_.find(id);
        require(it != workers_.end(), "unknown scheduler worker");
        return *it->second;
    }
    void fund_parents() {
        for (auto it = pending_parents_.begin(); it != pending_parents_.end();) {
            if (funded_ == tickets_ || inf_parents_ == preparation_window_) break;
            auto &p = parent(std::get<2>(*it));
            // The complete immutable inventory exists before the owner's grant.
            if (p.descriptor_only && p.submitted != p.expected) { ++it; continue; }
            // A guarded SAM restore can take an explicit singleton exception.
            // Its full physical backing remains charged even above the usual
            // ceiling. While in debt, ordinary and zero-byte grants both wait.
            bool singleton_restore = p.allow_oversize_host && p.bytes > capacity_ && !funded_ && !used_ &&
                                     !p.admission_reserve;
            const Id free = capacity_ - used_;
            if ((!singleton_restore && (p.admission_reserve > free ||
                 p.bytes + p.headroom > free - p.admission_reserve)) ||
                (inf_parents_ > 0 && p.bytes > inference_capacity_ - inf_bytes_)) {
                ++it;
                continue;
            }
            if (p.exclusive_kind) {
                auto available = std::find_if(workers_.begin(), workers_.end(), [&](const auto &entry) {
                    const auto &w = *entry.second;
                    return (p.exclusive_kind & w.kinds) && w.resident < 0 && !w.unsettled && !w.closing;
                });
                if (available == workers_.end()) { ++it; continue; }
                p.worker = available->first;
                available->second->resident = p.id;
            }
            p.funded = true;
            ++funded_;
            ++inf_parents_;
            used_ += p.bytes + p.headroom;
            inf_bytes_ += p.bytes;
            if (p.descriptor_only) {
                const Id granted = now_ns();
                for (const auto &item : p.planned_tasks) task(std::get<2>(item)).ready_since_ns = granted;
                ready_.merge(p.planned_tasks);
            } else {
                preparation_.emplace_back(p.id, p.generation);
            }
            it = pending_parents_.erase(it);
        }
    }
    static Id unreserved(const Parent &p) {
        return std::max<Id>(0, p.task_bytes - p.headroom);
    }
    bool fits(const Task &t, const Parent &p) const {
        Id remaining_escrow = std::max<Id>(0, p.headroom - p.task_bytes);
        return t.bytes <= remaining_escrow || t.bytes - remaining_escrow <= capacity_ - used_;
    }
    Worker *pick_worker(const Task &t, const Parent &p) {
        Worker *best = nullptr;
        for (auto &entry : workers_) {
            auto &w = *entry.second;
            if (w.closing || !(t.kinds & w.kinds) || w.active >= w.window ||
                (p.worker >= 0 && w.id != p.worker) ||
                (w.resident >= 0 && w.resident != p.id) ||
                (t.required_worker >= 0 && w.id != t.required_worker) ||
                !stages_.can_infer(w.device)) continue;
            if (std::any_of(t.handles.begin(), t.handles.end(), [&](const DetectorHandle &h) {
                return h.persistent && w.pending_sources.count(h.identity);
            })) continue;
            if (!best || std::tie(w.active, w.last_use, w.id) <
                         std::tie(best->active, best->last_use, best->id)) best = &w;
        }
        return best;
    }
    void dispatch_ready() {
        for (auto it = ready_.begin(); it != ready_.end();) {
            auto &t = task(std::get<2>(*it));
            auto &p = parent(t.parent);
            Worker *w = fits(t, p) ? pick_worker(t, p) : nullptr;
            if (!w) { ++it; continue; }
            Id old = unreserved(p);
            p.task_bytes += t.bytes;
            used_ += unreserved(p) - old;
            t.worker = w->id;
            t.generation = next_generation();
            stages_.begin_infer(w->device);
            ++w->active;
            ++w->unsettled;
            w->last_use = ++issued_;
            const Id current = now_ns();
            queue_ns_ += current - t.ready_since_ns;
            if (last_receipt_ns_) {
                ++receipt_refills_;
                receipt_refill_ns_ += current - last_receipt_ns_;
                last_receipt_ns_ = 0;
            }
            Lease lease{run_, p.id, t.id, w->id, w->epoch, w->pid, t.generation};
            OutgoingFiles files;
            for (size_t i = 0; i < t.handles.size(); ++i) {
                auto &handle = t.handles[i];
                if (handle.persistent && w->installed_sources.count(handle.identity)) continue;
                if (handle.persistent) w->pending_sources.insert(handle.identity);
                files.emplace_back(static_cast<std::uint32_t>(i), handle.file);
            }
            w->commands.push_back({dispatch, lease, std::move(t.data), std::move(files)});
            it = ready_.erase(it);
        }
    }
    void update_stage_policy() {
        bool gpu_backlog = std::any_of(ready_.begin(), ready_.end(), [&](const auto &entry) {
            const auto &t = task(std::get<2>(entry));
            return (t.kinds & 2) && fits(t, parent(t.parent));
        });
        if (gpu_backlog != last_backlog_) {
            stages_.apply("backlog", {gpu_backlog});
            last_backlog_ = gpu_backlog;
        }
    }
    void update_retirement_pressure() {
        // Count only science-adopted dense planes. File-result and seam escrow
        // can coexist in the same parent ticket but do not signal pressure.
        // Preserve the existing 75% / 62.5% hysteresis, without multiplying
        // a potentially large int64 capacity by four or eight.
        const Id low_water = (pressure_capacity_ / 8) * 5 + ((pressure_capacity_ % 8) * 5) / 8;
        const Id high_water = pressure_capacity_ - pressure_capacity_ / 4;
        bool pressure = retained_parent_bytes_ > 0 && pressure_capacity_ > 0 &&
                        (last_pressure_ ? retained_parent_bytes_ > low_water : retained_parent_bytes_ >= high_water);
        if (pressure != last_pressure_) {
            stages_.apply("pressure", {pressure});
            last_pressure_ = pressure;
        }
    }
    void dispatch_loop() noexcept {
        Lock lock(mutex_);
        try {
            for (;;) {
                auto ready = [&] { return dirty_ || stopping_ || !fault_.empty(); };
                auto deadline = stages_.next_deadline_ns();
                if (deadline > 0) {
                    changed_.wait_until(lock, std::chrono::steady_clock::time_point(
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::nanoseconds(deadline))), ready);
                } else {
                    changed_.wait(lock, ready);
                }
                if (stopping_ || !fault_.empty()) return;
                dirty_ = false;
                stages_.refresh_deadlines();
                fund_parents();
                update_stage_policy();
                dispatch_ready();
                update_stage_policy();
                last_receipt_ns_ = 0;
                changed_.notify_all();
            }
        } catch (const std::exception &e) { fail(e.what()); }
    }
    void writer_loop(Worker &w) noexcept {
        try {
            for (;;) {
                Packet out{};
                {
                    Lock lock(mutex_);
                    changed_.wait(lock, [&] { return !w.commands.empty() || stopping_ || !fault_.empty(); });
                    if (stopping_ || !fault_.empty()) return;
                    out = std::move(w.commands.front());
                    w.commands.pop_front();
                }
                send_packet(w.socket, out, stop_io_);
            }
        } catch (const std::exception &e) {
            Lock lock(mutex_);
            if (!stopping_) fail("worker " + std::to_string(w.id) + ": " + e.what());
        }
    }
    void validate(const Packet &p, const Worker &w) {
        const auto &l = p.lease;
        require(p.kind >= installed && p.kind <= control_reply, "invalid worker receipt type");
        require(l.run == run_ && l.worker == w.id && l.pid == w.pid && l.epoch == w.epoch,
                "receipt sender/run/worker epoch mismatch");
        if (p.kind == control_reply) {
            require(l.parent == 0 && l.task == 0 && w.outstanding_controls.count(l.generation),
                    "unrequested or duplicate worker control response");
            return;
        }
        auto &t = task(l.task);
        require(t.worker == w.id && t.generation && t.generation == l.generation && t.parent == l.parent,
                "receipt parent/task/lease generation mismatch");
        if (p.kind == installed) require(!t.acknowledged, "duplicate install receipt");
        if (p.kind == compute || p.kind == final) require(t.acknowledged, "completion before install ACK");
        if (p.kind == compute) require(!t.compute_seen, "duplicate compute receipt");
        if (p.kind == final) require(!t.final_seen, "duplicate final receipt");
    }
    void accept(Packet p, Worker &w) {
        validate(p, w);
        if (p.kind == control_reply) {
            w.outstanding_controls.erase(p.lease.generation);
            events_.push_back(std::move(p));
            wake();
            return;
        }
        auto &t = task(p.lease.task);
        if (p.kind == failed) {
            throw std::runtime_error("task " + std::to_string(t.id) + " failed: " + p.data.substr(0, 8192));
        }
        if (p.kind == installed) {
            for (auto &handle : t.handles) {
                if (handle.persistent && w.pending_sources.erase(handle.identity))
                    w.installed_sources.insert(handle.identity);
            }
            t.handles.clear();
            t.acknowledged = true;
            ++installed_;
        } else {
            if (!t.window_returned) {
                stages_.end_infer(w.device);
                last_receipt_ns_ = now_ns();
                t.window_returned = true;
                --w.active;
            }
            if (p.kind == compute) t.compute_seen = true;
            if (p.kind == final) {
                t.final_seen = true;
                ++finals_;
                --w.unsettled;
                auto &parent_state = parent(t.parent);
                if (!p.data.empty()) ++parent_state.unread_finals;
                if (++parent_state.completed == parent_state.expected) {
                    parent_state.published = true;
                    --inf_parents_;
                    inf_bytes_ -= parent_state.bytes;
                    if (parent_state.worker >= 0) worker(parent_state.worker).resident = -1;
                    publication_.emplace_back(t.parent, parent_state.generation);
                }
            }
        }
        if (!p.data.empty()) events_.push_back(std::move(p));
        wake();
    }
    void reader_loop(Worker &w) noexcept {
        try {
            for (;;) {
                auto frame = read_header(w.socket, stop_io_);
                auto &packet = frame.first;
                require(packet.kind != installed || frame.second == 0, "unexpected install receipt payload");
                {
                    Lock lock(mutex_);
                    if (stopping_ || !fault_.empty()) return;
                    validate(packet, w); // Reject forged lengths/identities before allocation.
                }
                packet.data.resize(static_cast<size_t>(frame.second));
                transfer(w.socket, packet.data.data(), packet.data.size(), false, stop_io_);
                Lock lock(mutex_);
                if (stopping_ || !fault_.empty()) return;
                accept(std::move(packet), w);
            }
        } catch (const std::exception &e) {
            Lock lock(mutex_);
            if (!stopping_ && !w.closing) fail("worker " + std::to_string(w.id) + ": " + e.what());
        }
    }
    template<class Predicate> bool wait(Lock &lock, Id ms, Predicate ready) {
        auto done = [&] { return ready() || stopping_ || !fault_.empty(); };
        if (ms < 0) changed_.wait(lock, done);
        else changed_.wait_for(lock, std::chrono::milliseconds(ms), done);
        healthy();
        return ready();
    }
public:
    Scheduler(Id run, Id capacity, Id tickets, Id inf_capacity, Id prep_window)
        : run_(run), capacity_(capacity), tickets_(tickets),
          inference_capacity_(inf_capacity), preparation_window_(prep_window), pressure_capacity_(capacity) {
        require(run > 0 && capacity >= 0 && tickets > 0 && inf_capacity >= 0 &&
                inf_capacity <= capacity && prep_window > 0, "invalid scheduler capacities or run ID");
        dispatcher_ = std::thread(&Scheduler::dispatch_loop, this);
    }
    ~Scheduler() { close_run(); }
    void configure(Id bytes, Id tickets, Id inference_bytes, Id preparation_window, Id pressure_capacity) {
        Lock lock(mutex_);
        healthy();
        require(parents_.empty(), "cannot reconfigure capacity after declaring work");
        require(bytes >= 0 && tickets > 0 && inference_bytes >= 0 && inference_bytes <= bytes &&
                preparation_window > 0 && pressure_capacity >= bytes, "invalid scheduler capacities");
        capacity_ = bytes;
        tickets_ = tickets;
        inference_capacity_ = inference_bytes;
        preparation_window_ = preparation_window;
        pressure_capacity_ = pressure_capacity;
    }
    bool try_resize_capacity(Id bytes, Id inference_bytes, Id pressure_capacity) {
        Lock lock(mutex_);
        healthy();
        require(bytes >= 0 && inference_bytes >= 0 && inference_bytes <= bytes && pressure_capacity >= bytes,
                "invalid dense repartition capacity");
        // This transfers actual capacity to another allocator. A merely
        // requested release, or an omitted pending restore, cannot fund it.
        if (used_ > bytes) return false;
        for (const auto &entry : parents_) {
            const auto &p = entry.second;
            if (p.retired || p.published) continue;
            if (p.expected || p.bytes > bytes || (!p.funded && p.admission_reserve > bytes - p.bytes)) return false;
        }
        capacity_ = bytes;
        inference_capacity_ = inference_bytes;
        pressure_capacity_ = pressure_capacity;
        update_retirement_pressure();
        wake();
        return true;
    }
    void add_worker(Id id, Id epoch, Id pid, Id kinds, Id window, Socket socket, Id device) {
        auto owned = std::make_unique<Worker>(id, epoch, pid, kinds, window, device, socket);
        Lock lock(mutex_);
        healthy();
        require(id >= 0 && epoch > 0 && pid > 0 && kinds > 0 && kinds <= 3 &&
                window > 0 && device >= -1 && socket != invalid_socket, "invalid worker registration");
        require(!workers_.count(id), "duplicate worker registration");
        auto &w = *owned;
        workers_.emplace(id, std::move(owned));
        try {
            w.reader = std::thread(&Scheduler::reader_loop, this, std::ref(w));
            w.writer = std::thread(&Scheduler::writer_loop, this, std::ref(w));
        } catch (const std::exception &e) { fail(e.what()); throw; }
        wake();
    }
    Id declare_parent(Id id, Id expected, Id bytes, Id priority, Id exclusive, Id headroom,
                      bool allow_oversize_host, Id pressure_bytes, bool descriptor_only, Id admission_reserve) {
        Lock lock(mutex_);
        healthy();
        require(id >= 0 && expected >= 0 && bytes >= 0 && headroom >= 0 && exclusive >= 0 && exclusive <= 3 &&
                (expected > 0 || (!exclusive && !headroom)),
                "invalid parent declaration");
        require(!allow_oversize_host || expected == 0, "oversize grant restricted to zero-task host restore");
        require(!descriptor_only || expected > 0, "immutable descriptor inventory requires scientific tasks");
        require(admission_reserve >= 0 && (!admission_reserve || expected == 0),
                "unallocated restore headroom requires a host parent");
        require(pressure_bytes >= 0 && pressure_bytes <= bytes, "invalid postprocess dense pressure bytes");
        require(allow_oversize_host || (bytes <= capacity_ && headroom <= capacity_ - bytes),
                "parent exceeds scheduler capacity");
        require(!exclusive || std::any_of(workers_.begin(), workers_.end(), [&](const auto &entry) {
            return (entry.second->kinds & exclusive) && !entry.second->closing;
        }), "exclusive parent has no compatible worker");
        require(!parents_.count(id), "duplicate parent identity");
        Id generation = next_generation();
        parents_.emplace(id, Parent{id, expected, bytes, priority, exclusive, headroom, allow_oversize_host,
                                    pressure_bytes, descriptor_only, admission_reserve, generation});
        pending_parents_.emplace(priority, ++sequence_, id);
        wake();
        return generation;
    }
    bool update_host_reserve(Id id, Id generation, Id reserve) {
        Lock lock(mutex_);
        healthy();
        auto &p = parent(id);
        require(!p.expected && p.generation == generation && reserve >= 0 && !p.retired,
                "invalid host reservation update");
        if (p.funded) return false;
        if (p.admission_reserve != reserve) {
            p.admission_reserve = reserve;
            wake();
        }
        return true;
    }
    std::pair<Id, Id> next_parent(bool publication, Id ms) {
        Lock lock(mutex_);
        auto &queue = publication ? publication_ : preparation_;
        if (!wait(lock, ms, [&] { return !queue.empty(); })) return {-1, 0};
        auto result = queue.front();
        queue.pop_front();
        if (!publication) parent(result.first).preparing = true;
        return result;
    }
    DetectorHandle pin_detector_file(int fd, const std::string &identity, int role) {
        return detector_files_.pin(fd, identity, role);
    }
    void submit(Id id, Id pid, Bytes data, Id kinds, Id priority, Id required, Id bytes, DetectorHandles handles,
                bool planned = false) {
        Lock lock(mutex_);
        healthy();
        auto &p = parent(pid);
        require(p.descriptor_only == planned && (planned || p.preparing) && !p.retired && p.submitted < p.expected,
                "parent lacks an available preparation or descriptor slot");
        require(!planned || std::all_of(handles.begin(), handles.end(),
                [](const DetectorHandle &handle) { return handle.persistent; }),
                "predeclared descriptors only accept immutable source handles");
        require(!planned || !p.exclusive_kind || (required < 0 && !(p.exclusive_kind & ~kinds)),
                "predeclared task incompatible with parent worker reservation");
        require(id >= 0 && kinds > 0 && kinds <= 3 && required >= -1 && bytes >= 0 &&
                bytes <= capacity_ - p.bytes && data.size() <= max_payload, "invalid prepared task descriptor");
        require(!tasks_.count(id), "duplicate task identity");
        if (required >= 0) {
            auto &w = worker(required);
            require((w.kinds & kinds) && (p.worker < 0 || p.worker == required), "required worker incompatible with task");
        }
        if (p.worker >= 0) require(worker(p.worker).kinds & kinds, "resident worker incompatible with task");
        require(std::any_of(workers_.begin(), workers_.end(), [&](const auto &entry) {
            return (entry.second->kinds & kinds) && !entry.second->closing;
        }), "prepared task has no compatible worker");
        auto inserted = tasks_.emplace(id, Task{id, pid, kinds, priority, required, bytes, std::move(data), now_ns()});
        inserted.first->second.handles = std::move(handles);
        ++p.submitted;
        auto &queue = planned ? p.planned_tasks : ready_;
        queue.emplace(priority, ++sequence_, id);
        wake();
    }
    bool wait_final(Id id, Id generation, Id ms) {
        Lock lock(mutex_);
        auto &t = task(id);
        require(generation > 0 && t.generation == generation, "task wait lease generation mismatch");
        return wait(lock, ms, [&] { return t.final_seen; });
    }
    void host_prepared(Id id, Id generation) {
        Lock lock(mutex_);
        healthy();
        auto &p = parent(id);
        require(!p.expected && p.preparing && !p.published && generation == p.generation,
                "invalid host preparation receipt");
        p.published = true;
        --inf_parents_;
        inf_bytes_ -= p.bytes;
        p.postprocess_adopted = true;
        retained_parent_bytes_ += p.pressure_bytes;
        update_retirement_pressure();
        publication_.emplace_back(id, generation);
        wake();
    }
    void adopt_postprocess(Id id, Id generation) {
        Lock lock(mutex_);
        healthy();
        auto &p = parent(id);
        require(p.expected && p.generation == generation && p.published && !p.retired && !p.postprocess_adopted,
                "invalid postprocess adoption receipt");
        p.postprocess_adopted = true;
        retained_parent_bytes_ += p.pressure_bytes;
        update_retirement_pressure();
        wake();
    }
    bool next_event(Packet &out, Id ms) {
        Lock lock(mutex_);
        if (!wait(lock, ms, [&] { return !events_.empty(); })) return false;
        out = std::move(events_.front());
        events_.pop_front();
        if (out.kind == final) --parent(out.lease.parent).unread_finals;
        return true;
    }
    void retire_task(Id id, Id generation) {
        Lock lock(mutex_);
        healthy();
        auto &t = task(id);
        require(generation > 0 && t.generation == generation && t.final_seen && !t.retired, "invalid task retirement receipt");
        auto &p = parent(t.parent);
        Id old = unreserved(p);
        p.task_bytes -= t.bytes;
        used_ -= old - unreserved(p);
        t.retired = true;
        wake();
    }
    void set_parent_remaining(Id id, Id generation, Id remaining) {
        Lock lock(mutex_);
        healthy();
        auto &p = parent(id);
        require(p.published && !p.retired && p.generation == generation && remaining >= 0 && remaining < p.bytes,
                "invalid parent component retirement receipt");
        used_ -= p.bytes - remaining;
        const Id dense_remaining = std::min(p.pressure_bytes, remaining);
        if (p.postprocess_adopted) retained_parent_bytes_ -= p.pressure_bytes - dense_remaining;
        p.pressure_bytes = dense_remaining;
        p.bytes = remaining;
        update_retirement_pressure();
        wake();
    }
    void retire_parent(Id id, Id generation) {
        Lock lock(mutex_);
        healthy();
        auto &p = parent(id);
        require(p.generation == generation && p.published && !p.retired && !p.task_bytes && !p.unread_finals,
                "invalid parent retirement or retained task aliases");
        used_ -= p.bytes + p.headroom;
        if (p.postprocess_adopted) retained_parent_bytes_ -= p.pressure_bytes;
        update_retirement_pressure();
        --funded_;
        p.retired = true;
        auto entry = std::find(publication_.begin(), publication_.end(), std::make_pair(id, generation));
        if (entry != publication_.end()) publication_.erase(entry);
        wake();
    }
    Id send_control(Id id, Bytes data, bool last, bool expect_reply) {
        Lock lock(mutex_);
        healthy();
        auto &w = worker(id);
        require(!w.closing && data.size() <= max_payload && w.commands.size() < static_cast<size_t>(w.window) + 64 &&
                w.outstanding_controls.size() < 64, "closed worker, full control lane or oversized control");
        if (last) {
            require(!expect_reply && w.outstanding_controls.empty() && !w.unsettled && ready_.empty() && pending_parents_.empty() &&
                    inf_parents_ == 0, "worker stop with outstanding inference work");
            w.closing = true;
        }
        Id generation = expect_reply ? next_generation() : 0;
        if (expect_reply) w.outstanding_controls.insert(generation);
        w.commands.push_back({control, {run_, 0, 0, id, w.epoch, w.pid, generation}, std::move(data)});
        changed_.notify_all();
        return generation;
    }
    void cancel(const std::string &reason) {
        Lock lock(mutex_);
        fail("native scheduler cancelled: " + reason);
    }
    std::vector<Id> stage(const std::string &op, const std::vector<Id> &fields) {
        Lock lock(mutex_);
        // Actors can finish an already-owned device fence after an unrelated
        // failure. Only proof/teardown operations survive a fault.
        static const std::set<std::string> fault_cleanup{
            "snapshot", "run_epoch", "release", "release_exclusive", "release_overlap",
            "release_resident", "quarantine", "cancel_prepare", "aux_complete", "aux_revoke", "cancel_request"
        };
        if (stopping_ || !fault_cleanup.count(op)) healthy();
        auto epoch = stages_.epoch();
        auto result = stages_.apply(op, fields);
        if (stages_.epoch() != epoch) wake();
        return result;
    }
    Id wait_stage_change(Id epoch, Id ms) {
        Lock lock(mutex_);
        wait(lock, ms, [&] { return stages_.epoch() != epoch; });
        return stages_.epoch();
    }
    Summary snapshot() const {
        Lock lock(mutex_);
        Id active = 0;
        for (const auto &entry : workers_) active += entry.second->active;
        return {{"bytes_used", used_}, {"byte_capacity", capacity_}, {"inference_byte_capacity", inference_capacity_},
                {"oversize_host_bytes", std::max<Id>(0, used_ - capacity_)}, {"tickets_used", funded_},
                {"retained_parent_bytes", retained_parent_bytes_}, {"retirement_pressure", last_pressure_},
                {"pressure_byte_capacity", pressure_capacity_},
                {"ticket_capacity", tickets_}, {"inference_bytes", inf_bytes_}, {"preparing_parents", inf_parents_},
                {"blocked_parents", static_cast<Id>(pending_parents_.size())},
                {"ready", static_cast<Id>(ready_.size())}, {"active", active}, {"dispatched", issued_},
                {"installed", installed_}, {"final", finals_}, {"events", static_cast<Id>(events_.size())},
                {"publication_tickets", static_cast<Id>(publication_.size())},
                {"dispatch_queue_ns", queue_ns_}, {"dispatch_queue_samples", issued_},
                {"receipt_refill_ns", receipt_refill_ns_}, {"receipt_refill_samples", receipt_refills_},
                {"stage_claims", stages_.claim_count()}, {"stage_epoch", stages_.epoch()}, {"failed", !fault_.empty()}};
    }
    void close_run() {
        // The binding serializes close/destroy calls. All other methods may be waiting.
        {
            Lock lock(mutex_);
            if (stopping_) return;
            stopping_ = true;
            stop_io_.signal();
            discard_detector_files();
            changed_.notify_all();
            for (auto &entry : workers_) interrupt_socket(entry.second->socket);
        }
        if (dispatcher_.joinable()) dispatcher_.join();
        for (auto &entry : workers_) {
            auto &w = *entry.second;
            if (w.writer.joinable()) w.writer.join();
            if (w.reader.joinable()) w.reader.join();
        }
        Lock lock(mutex_);
        for (auto &entry : workers_) {
            close_socket(entry.second->socket);
            entry.second->socket = invalid_socket;
        }
    }
};
} // namespace xta_scheduler
