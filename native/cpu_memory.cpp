#include "cpu_kernels.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace xta::cpu {
namespace {

using Path = std::filesystem::path;

// Kernel reads may block indefinitely on a virtual filesystem. The original
// Python thread state is used only for a signal, cold discovery or unusual
// locale text; ordinary readings take the GIL back once at the end.
class Sampling {
    PyThreadState* state_ = PyEval_SaveThread();
public:
    ~Sampling() { PyEval_RestoreThread(state_); }
    template<class F> auto with_python(F&& fn) {
        PyEval_RestoreThread(state_);
        struct Unlock {
            PyThreadState*& state;
            ~Unlock() { state = PyEval_SaveThread(); }
        } unlock{state_};
        return fn();
    }
};

std::optional<Path> file_path(PyObject* value) {
    Object p(PyOS_FSPath(value));
#ifdef _WIN32
    Object u(PyUnicode_Check(p.get()) ? Py_NewRef(p.get()) :
             PyUnicode_DecodeFSDefaultAndSize(PyBytes_AS_STRING(p.get()), PyBytes_GET_SIZE(p.get())));
    Py_ssize_t n = 0;
    wchar_t* raw = PyUnicode_AsWideCharString(u.get(), &n);
    if (!raw) throw PythonError{};
    struct Free { wchar_t* p; ~Free() { PyMem_Free(p); } } hold{raw};
    std::wstring name(raw, static_cast<std::size_t>(n));
#else
    PyObject* raw = PyBytes_Check(p.get()) ? Py_NewRef(p.get()) : PyUnicode_EncodeFSDefault(p.get());
    if (!raw) {
        // A process may change its text locale after startup, while its
        // filesystem encoding stays fixed. Fail at this path's ordered read.
        if (!PyErr_ExceptionMatches(PyExc_UnicodeEncodeError)) throw PythonError{};
        PyErr_Clear();
        return std::nullopt;
    }
    Object bytes(raw);
    std::string name(PyBytes_AS_STRING(bytes.get()), static_cast<std::size_t>(PyBytes_GET_SIZE(bytes.get())));
#endif
    return Path(name);
}

struct Read {
    int error = 0;
    std::string data;
};

Read read_file(const Path& path, Sampling& sampling) {
    Read result;
    // A malformed membership can contain an embedded NUL on virtual procfs.
    // Report the failure at the ordered read rather than opening its prefix.
    if (path.native().find(Path::value_type{}) != Path::string_type::npos) {
        result.error = EINVAL;
        return result;
    }
    int fd;
    auto interrupted = [&] {
        if (errno != EINTR) return false;
        sampling.with_python([] { if (PyErr_CheckSignals() < 0) throw PythonError{}; });
        return true;
    };
    do {
#ifdef _WIN32
        fd = _wopen(path.c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
#else
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    } while (fd < 0 && interrupted());
    if (fd < 0) { result.error = errno; return result; }
    struct Close {
        int fd;
        ~Close() {
#ifdef _WIN32
            _close(fd);
#else
            ::close(fd);
#endif
        }
    } close{fd};
    std::array<char, 16384> buffer;
    while (true) {
#ifdef _WIN32
        const auto n = _read(fd, buffer.data(), static_cast<unsigned>(buffer.size()));
#else
        const auto n = ::read(fd, buffer.data(), buffer.size());
#endif
        if (n == 0) return result;
        if (n < 0) {
            if (interrupted()) continue;
            result.error = errno;
            result.data.clear();
            return result;
        }
        result.data.append(buffer.data(), static_cast<std::size_t>(n));
    }
}

struct Group {
    bool v2, mount;
    std::vector<std::optional<Path>> nodes;
};

std::vector<Group> parse_plan(PyObject* value) {
    Object plan(PySequence_Fast(value, "Expected memory sample groups"));
    std::vector<Group> groups;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(plan.get()); ++i) {
        auto item = PySequence_Fast_GET_ITEM(plan.get(), i);
        PyObject* dirs;
        int v2, mount;
        if (!PyArg_ParseTuple(item, "ppO:memory_group", &v2, &mount, &dirs)) throw PythonError{};
        Object nodes(PySequence_Fast(dirs, "Expected cgroup ancestor paths"));
        Group group{v2 != 0, mount != 0, {}};
        for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(nodes.get()); ++j)
            group.nodes.emplace_back(file_path(PySequence_Fast_GET_ITEM(nodes.get(), j)));
        groups.push_back(std::move(group));
    }
    return groups;
}

enum class Counter { invalid, finite, one, unlimited };

// All kernel counter formats are ASCII. For unusual locale text apply the same
// Unicode decoding/strip rules as Path.read_text, without rereading the file.
std::optional<std::string> unusual_counter(const Read& input, const char* encoding) {
    Object text;
    PyObject* decoded = PyUnicode_Decode(input.data.data(), input.data.size(), encoding, "strict");
    if (!decoded) {
        if (!PyErr_ExceptionMatches(PyExc_UnicodeError)) throw PythonError{};
        PyErr_Clear();
        return std::nullopt;
    }
    text = Object(decoded);
    Object stripped(PyObject_CallMethod(text.get(), "strip", nullptr));
    if (!PyUnicode_IS_ASCII(stripped.get())) return std::nullopt;
    Py_ssize_t n = 0;
    const char* raw = PyUnicode_AsUTF8AndSize(stripped.get(), &n);
    if (!raw) throw PythonError{};
    return std::string(raw, static_cast<std::size_t>(n));
}

Counter classify(const Read& input, bool limit, bool v2, const char* encoding,
                 int max_digits, Sampling& sampling) {
    if (input.error) return Counter::invalid;
    std::string_view field(input.data);
    std::optional<std::string> decoded;
    if (std::any_of(field.begin(), field.end(), [](unsigned char c) { return c >= 128 || c == 0; })) {
        decoded = sampling.with_python([&] { return unusual_counter(input, encoding); });
        if (!decoded) return Counter::invalid;
        field = *decoded;
    } else {
        constexpr std::string_view spaces = " \t\n\v\f\r\x1c\x1d\x1e\x1f";
        auto first = field.find_first_not_of(spaces);
        if (first == std::string_view::npos) return Counter::invalid;
        field = field.substr(first, field.find_last_not_of(spaces) - first + 1);
    }
    if (limit && v2 && field == "max") return Counter::unlimited;
    if (field.empty() || (max_digits > 0 && field.size() > static_cast<std::size_t>(max_digits))
            || field.find_first_not_of("0123456789") != std::string_view::npos) return Counter::invalid;
    const auto first = field.find_first_not_of('0');
    if (first == std::string_view::npos) return Counter::finite;
    field.remove_prefix(first);
    constexpr std::string_view v1_sentinel = "1152921504606846976";
    if (limit && !v2 && (field.size() > v1_sentinel.size()
            || (field.size() == v1_sentinel.size() && field >= v1_sentinel))) return Counter::unlimited;
    return field == "1" ? Counter::one : Counter::finite;
}

struct Level { Read limit, usage, stat; bool hierarchical; };
struct Observations { std::optional<Read> mount; std::vector<Level> levels; };

bool mount_text_readable(const Read& input, const char* encoding, Sampling& sampling) {
    if (input.error) return false;
    return sampling.with_python([&] {
        PyObject* text = PyUnicode_Decode(input.data.data(), input.data.size(), encoding, "strict");
        if (text) {
            Py_DECREF(text);
            return true;
        }
        if (!PyErr_ExceptionMatches(PyExc_UnicodeError)) throw PythonError{};
        PyErr_Clear();
        return false;
    });
}

Object bytes_or_none(const Read& read) {
    return Object(read.error ? Py_NewRef(Py_None) :
                  PyBytes_FromStringAndSize(read.data.data(), static_cast<Py_ssize_t>(read.data.size())));
}

PyObject* sample(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *meminfo, *layout, *mount;
        const char* encoding;
        int digits;
        if (!PyArg_ParseTuple(args, "OOOsi:sample_memory_files", &meminfo, &layout, &mount, &encoding, &digits))
            return nullptr;
        const auto info_path = meminfo == Py_None ? std::nullopt : file_path(meminfo);
        const auto mount_path = file_path(mount);
        const bool cold = PyCallable_Check(layout) != 0;
        std::vector<Group> groups;
        if (!cold) groups = parse_plan(layout);
        Read info{ENOENT, {}};
        std::vector<Observations> observations;
        bool failed = false;
        {
            Sampling sampling;
            if (info_path) info = read_file(*info_path, sampling);
            if (cold) {
                groups = sampling.with_python([&] {
                    Object result(PyObject_CallNoArgs(layout));
                    return parse_plan(result.get());
                });
            }
            for (std::size_t index = 0; index < groups.size(); ++index) {
                const auto& group = groups[index];
                observations.emplace_back();
                auto& observation = observations.back();
                if (group.mount) {
                    // Repeated membership entries still take fresh pressure samples.
                    // A successful mount read for that same hierarchy is already known.
                    for (std::size_t k = 0; k < index; ++k) {
                        const auto& prior = groups[k];
                        const auto& old = observations[k].mount;
                        if (!group.nodes.empty() && !prior.nodes.empty() && prior.v2 == group.v2
                                && prior.nodes.back() == group.nodes.back() && old
                                && mount_text_readable(*old, encoding, sampling)) {
                            observation.mount = *old;
                            break;
                        }
                    }
                    if (!observation.mount)
                        observation.mount = mount_path ? read_file(*mount_path, sampling) : Read{EINVAL, {}};
                }
                for (const auto& directory : group.nodes) {
                    if (!directory) { failed = true; break; }
                    const auto& node = *directory;
                    Read cap = read_file(node / (group.v2 ? "memory.max" : "memory.limit_in_bytes"), sampling);
                    if (cap.error == ENOENT) continue;
                    const auto kind = classify(cap, true, group.v2, encoding, digits, sampling);
                    if (kind == Counter::unlimited) continue;
                    if (kind == Counter::invalid) { failed = true; break; }
                    Read usage = read_file(node / (group.v2 ? "memory.current" : "memory.usage_in_bytes"), sampling);
                    if (classify(usage, false, group.v2, encoding, digits, sampling) == Counter::invalid) {
                        failed = true;
                        break;
                    }
                    const bool hierarchical = group.v2 || classify(read_file(node / "memory.use_hierarchy", sampling),
                        false, false, encoding, digits, sampling) == Counter::one;
                    Read stat = hierarchical ? read_file(node / "memory.stat", sampling) : Read{ENOENT, {}};
                    observation.levels.push_back({std::move(cap), std::move(usage), std::move(stat), hierarchical});
                }
                if (failed) break;
            }
        }
        Object out(PyTuple_New(static_cast<Py_ssize_t>(observations.size())));
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const auto& group = observations[i];
            Object levels(PyTuple_New(static_cast<Py_ssize_t>(group.levels.size())));
            for (std::size_t j = 0; j < group.levels.size(); ++j) {
                const auto& level = group.levels[j];
                auto cap = bytes_or_none(level.limit), usage = bytes_or_none(level.usage), stat = bytes_or_none(level.stat);
                Object fields(Py_BuildValue("(OOOO)", cap.get(), usage.get(),
                                            level.hierarchical ? Py_True : Py_False, stat.get()));
                PyTuple_SET_ITEM(levels.get(), static_cast<Py_ssize_t>(j), fields.release());
            }
            Object mount_bytes(group.mount ? bytes_or_none(*group.mount).release() : Py_NewRef(Py_None));
            Object pair(PyTuple_Pack(2, mount_bytes.get(), levels.get()));
            PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), pair.release());
        }
        auto raw_info = bytes_or_none(info);
        return PyTuple_Pack(3, raw_info.get(), out.get(), failed ? Py_True : Py_False);
    });
}

PyMethodDef methods[] = {
    {"sample_memory_files", sample, METH_VARARGS, "Capture one ordered and fresh physical/cgroup memory observation."},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace
int register_memory(PyObject* module) { return PyModule_AddFunctions(module, methods); }
} // namespace xta::cpu
