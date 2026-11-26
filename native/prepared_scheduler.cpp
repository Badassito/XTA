// Thin CPython ownership and conversion boundary. No native thread uses Python.
#define PY_SSIZE_T_CLEAN
#include "prepared_scheduler.hpp"
#include <Python.h>

namespace {
using namespace xta_scheduler;
constexpr auto capsule_name = "XTA.prepared_scheduler.v1";

class Unlocked {
    PyThreadState *state_ = PyEval_SaveThread();
public:
    ~Unlocked() { PyEval_RestoreThread(state_); }
};

struct PythonError {};
class Args {
    PyObject *args_;
public:
    explicit Args(PyObject *args) : args_(args) {}
    PyObject *at(int i) const {
        auto *obj = PyTuple_GetItem(args_, i);
        if (!obj) throw PythonError{};
        return obj;
    }
    Id number(int i) const {
        auto value = PyLong_AsLongLong(at(i));
        if (PyErr_Occurred()) throw PythonError{};
        return value;
    }
    Bytes bytes(int i) const {
        char *data;
        Py_ssize_t size;
        if (PyBytes_AsStringAndSize(at(i), &data, &size) < 0) throw PythonError{};
        require(size <= max_payload, "scheduler payload exceeds transport bound");
        return {data, static_cast<size_t>(size)};
    }
    std::string string(int i) const {
        const char *value = PyUnicode_AsUTF8(at(i));
        if (!value) throw PythonError{};
        return value;
    }
    Scheduler &owner() const {
        auto *owner = static_cast<Scheduler *>(PyCapsule_GetPointer(at(0), capsule_name));
        if (!owner) throw PythonError{};
        return *owner;
    }
};
PyObject *none() { Py_RETURN_NONE; }
PyObject *number(Id value) { return PyLong_FromLongLong(value); }
PyObject *boolean(bool value) { return PyBool_FromLong(value); }

// Never pass a Python exception or a C++ exception across a foreign ABI.
template <PyObject *(*Function)(Args)> PyObject *call(PyObject *, PyObject *args) {
    try {
        return Function(Args(args));
    } catch (const PythonError &) {
        return nullptr;
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
void dispose(PyObject *capsule) {
    auto *owner = static_cast<Scheduler *>(PyCapsule_GetPointer(capsule, capsule_name));
    if (owner) {
        Unlocked unlock;
        delete owner;
    }
}
PyObject *create(Args a) {
    auto owner = std::make_unique<Scheduler>(a.number(0), a.number(1), a.number(2), a.number(3), a.number(4));
    PyObject *result = PyCapsule_New(owner.get(), capsule_name, dispose);
    if (result) owner.release();
    return result;
}
PyObject *configure(Args a) {
    a.owner().configure(a.number(1), a.number(2), a.number(3), a.number(4), a.number(5));
    return none();
}
PyObject *try_resize_capacity(Args a) {
    return boolean(a.owner().try_resize_capacity(a.number(1), a.number(2), a.number(3)));
}
PyObject *register_worker(Args a) {
    auto &owner = a.owner();
    Id id = a.number(1), epoch = a.number(2), pid = a.number(3), kinds = a.number(4);
    Id window = a.number(5), handle = a.number(6), device = a.number(7);
    require(handle >= 0, "invalid worker socket handle");
#ifdef _WIN32
    WSAPROTOCOL_INFOW info{};
    require(WSADuplicateSocketW(static_cast<Socket>(handle), GetCurrentProcessId(), &info) == 0,
            "failed to duplicate scheduler socket");
    auto socket = WSASocketW(FROM_PROTOCOL_INFO, FROM_PROTOCOL_INFO, FROM_PROTOCOL_INFO, &info, 0, WSA_FLAG_OVERLAPPED);
#else
    auto socket = dup(static_cast<Socket>(handle));
#endif
    require(socket != invalid_socket, "failed to acquire scheduler socket");
    owner.add_worker(id, epoch, pid, kinds, window, socket, device);
    return none();
}
PyObject *declare_parent(Args a) {
    return number(a.owner().declare_parent(a.number(1), a.number(2), a.number(3), a.number(4), a.number(5),
                                           a.number(6), bool(a.number(7)), a.number(8), bool(a.number(9)), a.number(10)));
}
PyObject *update_host_reserve(Args a) {
    return boolean(a.owner().update_host_reserve(a.number(1), a.number(2), a.number(3)));
}
PyObject *next_parent(Args a) {
    auto &owner = a.owner();
    bool publication = a.number(1) != 0;
    Id timeout = a.number(2);
    std::pair<Id, Id> result;
    { Unlocked unlock; result = owner.next_parent(publication, timeout); }
    if (result.first < 0) return none();
    return Py_BuildValue("(LL)", static_cast<long long>(result.first), static_cast<long long>(result.second));
}
PyObject *submit_packet(Args a, bool planned) {
    auto &owner = a.owner();
    auto *specs = a.at(8);
    require(PyTuple_Check(specs) && PyTuple_GET_SIZE(specs) <= max_payload,
            "invalid prepared detector handle table");
    DetectorHandles handles;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(specs); ++i) {
        auto *item = PyTuple_GET_ITEM(specs, i);
        int fd, role;
        const char *identity;
        if (!PyArg_ParseTuple(item, "isi", &fd, &identity, &role)) throw PythonError{};
        require(fd >= 0 && role >= 0 && role <= 2, "invalid prepared detector handle");
        require(!planned || role == 0, "planned detector work only accepts immutable source handles");
        handles.push_back(owner.pin_detector_file(fd, identity, role));
    }
    owner.submit(a.number(1), a.number(2), a.bytes(3), a.number(4), a.number(5), a.number(6), a.number(7),
                 std::move(handles), planned);
    return none();
}
PyObject *submit(Args a) { return submit_packet(a, false); }
PyObject *submit_planned(Args a) { return submit_packet(a, true); }
PyObject *next_event(Args a) {
    auto &owner = a.owner();
    Id ms = a.number(1);
    Packet packet{};
    bool ready;
    { Unlocked unlock; ready = owner.next_event(packet, ms); }
    if (!ready) return none();
    const auto &l = packet.lease;
    return Py_BuildValue("(L(LLLLLLL)y#)", static_cast<long long>(packet.kind),
                         static_cast<long long>(l.run), static_cast<long long>(l.parent),
                         static_cast<long long>(l.task), static_cast<long long>(l.worker),
                         static_cast<long long>(l.epoch), static_cast<long long>(l.pid),
                         static_cast<long long>(l.generation), packet.data.data(),
                         static_cast<Py_ssize_t>(packet.data.size()));
}
PyObject *wait_final(Args a) {
    auto &owner = a.owner();
    Id id = a.number(1), generation = a.number(2), timeout = a.number(3);
    bool result;
    { Unlocked unlock; result = owner.wait_final(id, generation, timeout); }
    return boolean(result);
}
PyObject *retire_task(Args a) { a.owner().retire_task(a.number(1), a.number(2)); return none(); }
PyObject *retire_parent(Args a) { a.owner().retire_parent(a.number(1), a.number(2)); return none(); }
PyObject *host_prepared(Args a) { a.owner().host_prepared(a.number(1), a.number(2)); return none(); }
PyObject *adopt_postprocess(Args a) { a.owner().adopt_postprocess(a.number(1), a.number(2)); return none(); }
PyObject *set_parent_remaining(Args a) {
    a.owner().set_parent_remaining(a.number(1), a.number(2), a.number(3));
    return none();
}
PyObject *send_control(Args a) {
    return number(a.owner().send_control(a.number(1), a.bytes(2), bool(a.number(3)), bool(a.number(4))));
}
PyObject *snapshot(Args a) {
    auto fields = a.owner().snapshot();
    auto *out = PyDict_New();
    if (!out) return nullptr;
    for (const auto &field : fields) {
        auto *value = number(field.second);
        if (!value || PyDict_SetItemString(out, field.first.c_str(), value) < 0) {
            Py_XDECREF(value);
            Py_DECREF(out);
            return nullptr;
        }
        Py_DECREF(value);
    }
    return out;
}
PyObject *stage(Args a) {
    auto &owner = a.owner();
    auto op = a.string(1);
    PyObject *fields = a.at(2);
    require(PyTuple_Check(fields), "device stage fields must be a tuple");
    std::vector<Id> data;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(fields); ++i) {
        auto value = PyLong_AsLongLong(PyTuple_GET_ITEM(fields, i));
        if (PyErr_Occurred()) throw PythonError{};
        data.push_back(value);
    }
    auto result = owner.stage(op, data);
    PyObject *out = PyTuple_New(static_cast<Py_ssize_t>(result.size()));
    if (!out) return nullptr;
    for (size_t i = 0; i < result.size(); ++i) {
        PyObject *value = number(result[i]);
        if (!value) { Py_DECREF(out); return nullptr; }
        PyTuple_SET_ITEM(out, static_cast<Py_ssize_t>(i), value);
    }
    return out;
}
PyObject *wait_stage_change(Args a) {
    auto &owner = a.owner();
    Id epoch = a.number(1), ms = a.number(2), result;
    { Unlocked unlock; result = owner.wait_stage_change(epoch, ms); }
    return number(result);
}
PyObject *cancel(Args a) { a.owner().cancel(a.string(1)); return none(); }
PyObject *close_run(Args a) {
    auto &owner = a.owner();
    { Unlocked unlock; owner.close_run(); }
    return none();
}
#define BIND(name) {#name, call<name>, METH_VARARGS, nullptr}
PyMethodDef methods[] = {
    BIND(create), BIND(configure), BIND(try_resize_capacity), BIND(register_worker), BIND(declare_parent), BIND(next_parent),
    BIND(submit), BIND(submit_planned), BIND(wait_final), BIND(next_event), BIND(retire_task), BIND(retire_parent),
    BIND(send_control), BIND(snapshot), BIND(stage), BIND(wait_stage_change), BIND(set_parent_remaining),
    BIND(host_prepared), BIND(adopt_postprocess), BIND(update_host_reserve), BIND(cancel), BIND(close_run),
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_prepared_scheduler",
                      "Native prepared-work scheduler and private actor transport.", -1, methods};
} // namespace
PyMODINIT_FUNC PyInit__prepared_scheduler() {
    auto *result = PyModule_Create(&module);
    if (result && PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
