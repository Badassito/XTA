#pragma once

// Device-stage state is owned by PreparedWorkScheduler and accessed only while
// its mutex is held. No Python callback or second admission ledger participates.
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace xta_scheduler {

class DeviceStages {
    struct Device {
        Id inference = 0, stage = 0, stage_purpose = 0;
        Id provisional = 0, provisional_purpose = 0, provisional_resident = 0;
        Id overlap = 0, overlap_purpose = 0, overlap_budget = 0;
        Id resident = 0, resident_purpose = 0, quarantine_reason = 0;
        bool quarantined = false;
        Id retired_control = 0, aux_offer = 0, aux_token = 0, aux_task = 0;
        Id burst = 0;
    };
    struct Request { Id purpose, expires, first; std::vector<Id> devices; };
    std::map<Id, Device> devices_;
    std::set<Id> workers_, overlap_declined_;
    std::map<Id, Id> purposes_;
    std::vector<Request> requests_;
    Id run_epoch_ = 0, change_epoch_ = 1, next_token_ = 1;
    Id reserved_ = -1, handoff_ = -1, handoff_deadline_ = 0, retry_after_ = 0;
    Id cursor_ = 0, aux_cursor_ = 0;
    Id overlap_acquisitions_ = 0, overlap_declines_ = 0, handoffs_ = 0;
    Id age_acquisitions_ = 0, pressure_acquisitions_ = 0, expirations_ = 0;
    bool priority_ = false, retiring_ = false, backlog_ = false, pressure_ = false;
    bool pressure_enabled_ = true, age_enabled_ = true;
    static constexpr Id second = 1000000000;
    Id expiry_delay_ = 30 * second, age_delay_ = 30 * second;
    Id handoff_delay_ = 2 * second, cooldown_delay_ = 10 * second;

    static void check(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static void arity(const std::vector<Id>& f, size_t n) {
        check(f.size() == n, "Invalid native device-stage operation arity");
    }
    static Id now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void changed() { ++change_epoch_; }
    Id token() { check(next_token_ > 0, "Native device-stage token overflow"); return next_token_++; }
    bool worker(Id device) const { return workers_.count(device) != 0; }
    Id kind(Id purpose) const {
        auto it = purposes_.find(purpose);
        return it == purposes_.end() ? 0 : it->second;
    }
    bool projection(Id purpose) const { return (kind(purpose) & 1) != 0; }
    bool sam(Id purpose) const { return (kind(purpose) & 2) != 0; }
    bool sibling_projection(Id purpose) const { return (kind(purpose) & 4) != 0; }
    bool any_inference() const {
        for (const auto& pair : devices_) if (pair.second.inference) return true;
        return false;
    }
    bool projection_active() const {
        for (const auto& pair : devices_)
            if (pair.second.stage && projection(pair.second.stage_purpose)) return true;
        return false;
    }
    bool any_live() const {
        for (const auto& pair : devices_) {
            const auto& d = pair.second;
            if (d.inference || d.stage || d.provisional || d.overlap || d.resident || d.aux_token) return true;
        }
        return false;
    }
    Request* next_request(Id device, Id timestamp) {
        for (auto& r : requests_)
            if ((pressure_ || (age_enabled_ && timestamp - r.first >= age_delay_)) &&
                std::find(r.devices.begin(), r.devices.end(), device) != r.devices.end()) return &r;
        return nullptr;
    }
    void clear_reservation() { reserved_ = handoff_ = -1; handoff_deadline_ = 0; }
    void erase_request(Id purpose) {
        requests_.erase(std::remove_if(requests_.begin(), requests_.end(),
            [purpose](const Request& r) { return r.purpose == purpose; }), requests_.end());
    }
    void refresh(Id timestamp, bool choose = true) {
        const auto old_size = requests_.size();
        requests_.erase(std::remove_if(requests_.begin(), requests_.end(),
            [timestamp](const Request& r) { return r.expires <= timestamp; }), requests_.end());
        if (old_size != requests_.size()) changed();
        if (requests_.empty() || !pressure_enabled_ || retiring_ || !backlog_ ||
            (reserved_ >= 0 && next_request(reserved_, timestamp) == nullptr)) {
            if (reserved_ >= 0 || handoff_ >= 0) { clear_reservation(); changed(); }
        } else if (handoff_ >= 0 && timestamp >= handoff_deadline_) {
            devices_[handoff_].burst = std::max<Id>(2, devices_[handoff_].burst);
            clear_reservation(); ++expirations_; changed();
        }
        if (!choose || reserved_ >= 0 || !backlog_ || retiring_ || !pressure_enabled_ ||
            timestamp < retry_after_ || projection_active()) return;
        std::vector<Id> available;
        for (Id id : workers_) {
            const auto& d = devices_[id];
            if (!d.stage && !d.overlap && !d.provisional && !d.aux_token && d.burst < 2)
                available.push_back(id);
        }
        std::vector<Id> candidates;
        for (const auto& r : requests_) {
            if (!pressure_ && !(age_enabled_ && timestamp - r.first >= age_delay_)) continue;
            for (Id id : r.devices)
                if (std::find(available.begin(), available.end(), id) != available.end()) candidates.push_back(id);
            if (!candidates.empty()) break;
        }
        if (candidates.empty()) return;
        std::vector<Id> ordered(workers_.begin(), workers_.end());
        const auto rank = [&](Id id) {
            Id position = std::find(ordered.begin(), ordered.end(), id) - ordered.begin();
            return (position - cursor_ + static_cast<Id>(ordered.size())) % static_cast<Id>(ordered.size());
        };
        reserved_ = *std::min_element(candidates.begin(), candidates.end(), [&](Id a, Id b) {
            if (devices_[a].inference != devices_[b].inference) return devices_[a].inference < devices_[b].inference;
            return rank(a) < rank(b);
        });
        cursor_ = (std::find(ordered.begin(), ordered.end(), reserved_) - ordered.begin() + 1) % ordered.size();
        changed();
    }
    bool eligible(Id device, Id purpose, Id provisional = 0, Id resident = 0) {
        if (device < 0) return false;
        refresh(now());
        const auto& d = devices_[device];
        if (resident && d.resident != resident) return false;
        if (d.resident && !resident && (d.quarantined || !projection(purpose))) return false;
        if (d.stage || d.overlap || d.aux_token ||
            (d.provisional && d.provisional != provisional) || d.inference) return false;
        // Quarantine forbids lending, not the owner's fenced shutdown work.
        if (resident) return true;
        if (retiring_ && worker(device)) return d.retired_control != 0 && sam(purpose);
        if (projection(purpose)) {
            if (!backlog_ || !worker(device)) return true;
            Request* r = next_request(device, now());
            return reserved_ == device && r && r->purpose == purpose && !projection_active();
        }
        return !(priority_ && worker(device));
    }
    Id prepare(Id device, Id purpose, Id epoch, Id resident) {
        if (epoch != run_epoch_ || !eligible(device, purpose, 0, resident)) return 0;
        auto& d = devices_[device];
        d.provisional = token(); d.provisional_purpose = purpose; d.provisional_resident = resident;
        d.aux_offer = 0; changed(); return d.provisional;
    }
    Id commit(Id device, Id purpose, Id epoch, Id reservation, Id resident) {
        auto& d = devices_[device];
        if (epoch != run_epoch_ || d.provisional != reservation || !reservation ||
            d.provisional_purpose != purpose || d.provisional_resident != resident ||
            !eligible(device, purpose, reservation, resident)) return 0;
        d.stage = reservation; d.stage_purpose = purpose;
        d.provisional = d.provisional_purpose = d.provisional_resident = 0;
        if (projection(purpose)) {
            if (worker(device)) {
                ++d.burst;
                if (backlog_ && pressure_) ++pressure_acquisitions_;
                else {
                    for (const auto& r : requests_)
                        if (r.purpose == purpose && backlog_ && now() - r.first >= age_delay_) ++age_acquisitions_;
                }
                if (backlog_ && handoff_ == device) ++handoffs_;
            }
            erase_request(purpose); clear_reservation();
        }
        changed(); return d.stage;
    }
    bool release(Id device, Id purpose, Id lease) {
        auto& d = devices_[device];
        if (!lease || d.stage != lease || d.stage_purpose != purpose) return false;
        d.stage = d.stage_purpose = 0;
        const Id timestamp = now();
        refresh(timestamp, false);
        if (projection(purpose) && reserved_ < 0 && !d.overlap && backlog_ && !retiring_ &&
            timestamp >= retry_after_ && pressure_enabled_ && !projection_active() && d.burst < 2 &&
            next_request(device, timestamp)) {
            reserved_ = handoff_ = device; handoff_deadline_ = timestamp + handoff_delay_;
        }
        changed(); return true;
    }
public:
    Id epoch() const { return change_epoch_; }
    void refresh_deadlines() { refresh(now()); }
    Id claim_count() const {
        Id total = 0;
        for (const auto& item : devices_) {
            const auto& d = item.second;
            total += d.inference + (d.stage != 0) + (d.provisional != 0) +
                     (d.overlap != 0) + (d.resident != 0) + (d.aux_token != 0);
        }
        return total;
    }
    Id next_deadline_ns() const {
        Id result = 0, timestamp = now();
        const auto take = [&](Id value) { if (value > timestamp && (!result || value < result)) result = value; };
        if (handoff_ >= 0) take(handoff_deadline_);
        for (const auto& r : requests_) {
            take(r.expires);
            if (age_enabled_ && backlog_ && pressure_enabled_ && !retiring_) take(r.first + age_delay_);
        }
        if (backlog_ && !requests_.empty()) take(retry_after_);
        return result;
    }
    bool can_infer(Id device) {
        if (device < 0) return true;
        refresh(now()); const auto& d = devices_[device];
        return !d.retired_control && !d.resident && !d.provisional && !d.stage &&
               !d.aux_token && reserved_ != device;
    }
    void begin_infer(Id device) {
        if (device < 0) return;
        check(can_infer(device), "Native device is not eligible for inference");
        ++devices_[device].inference; devices_[device].burst = 0; changed();
    }
    void end_infer(Id device) {
        if (device < 0) return;
        auto& d = devices_[device]; check(d.inference > 0, "Native inference stage credit underflow");
        --d.inference; changed(); refresh(now());
    }
    std::vector<Id> apply(const std::string& op, const std::vector<Id>& f) {
        refresh(now());
        if (op == "purpose") {
            arity(f, 2); check(f[0] > 0 && f[1] >= 0 && f[1] <= 7, "Invalid native stage purpose");
            auto it = purposes_.find(f[0]); check(it == purposes_.end() || it->second == f[1], "Conflicting stage purpose");
            purposes_[f[0]] = f[1]; return {};
        }
        if (op == "configure" || op == "reset") {
            check(!any_live(), "Cannot reset native device stages with live owners");
            if (op == "reset") arity(f, 0);
            for (Id id : f) check(id >= 0, "Negative configured GPU device");
            devices_.clear(); workers_.clear(); workers_.insert(f.begin(), f.end());
            for (Id id : workers_) devices_[id] = Device{};
            requests_.clear(); overlap_declined_.clear(); clear_reservation();
            priority_ = !workers_.empty(); retiring_ = backlog_ = pressure_ = false;
            retry_after_ = cursor_ = aux_cursor_ = 0;
            overlap_acquisitions_ = overlap_declines_ = handoffs_ = age_acquisitions_ = pressure_acquisitions_ = expirations_ = 0;
            ++run_epoch_; changed(); return {run_epoch_};
        }
        if (op == "run_epoch") { arity(f, 0); return {run_epoch_}; }
        if (op == "inference_active") { arity(f, 0); return {backlog_ || any_inference() ? 1 : 0}; }
        if (op == "floor_pending") {
            arity(f, 1);
            const Id timestamp = now();
            for (const auto& request : requests_) {
                if (request.purpose == f[0] || sibling_projection(request.purpose) || request.expires <= timestamp)
                    continue;
                if (!overlap_declined_.count(request.purpose) ||
                    (pressure_enabled_ && (pressure_ || (age_enabled_ && timestamp - request.first >= age_delay_))))
                    return {1};
            }
            return {0};
        }
        if (op == "timing") {
            arity(f, 4);
            check(!any_live() && requests_.empty(), "Cannot change native stage timing with live work");
            for (Id value : f) check(value > 0, "Native stage timing must be positive nanoseconds");
            expiry_delay_ = f[0]; age_delay_ = f[1]; handoff_delay_ = f[2]; cooldown_delay_ = f[3];
            changed(); return {};
        }
        if (op == "policy") {
            arity(f, 2); pressure_enabled_ = f[0] != 0; age_enabled_ = f[1] != 0; changed(); refresh(now()); return {};
        }
        if (op == "priority" || op == "retiring" || op == "backlog" || op == "pressure") {
            arity(f, 1); bool value = f[0] != 0;
            if (op == "priority") priority_ = value && !workers_.empty();
            if (op == "retiring") retiring_ = value;
            if (op == "backlog") backlog_ = value;
            if (op == "pressure") pressure_ = value;
            changed(); refresh(now()); return {};
        }
        if (op == "retired") {
            arity(f, 3); check(f[2] == run_epoch_ && worker(f[0]) && f[1] < 0, "Stale GPU asset-retirement proof");
            auto& d = devices_[f[0]];
            if (d.retired_control) { check(d.retired_control == f[1], "Conflicting GPU asset-retirement proof"); return {}; }
            check(retiring_ && !backlog_ && !any_inference(), "GPU asset retirement requires sealed global drain");
            d.retired_control = f[1]; changed(); return {};
        }
        if (op == "can_infer") { arity(f, 1); return {can_infer(f[0]) ? 1 : 0}; }
        if (op == "begin_infer") { arity(f, 1); if (!can_infer(f[0])) return {0}; begin_infer(f[0]); return {1}; }
        if (op == "end_infer") { arity(f, 1); end_infer(f[0]); return {}; }
        if (op == "can_aux") {
            arity(f, 1); const auto& d = devices_[f[0]];
            return {(!d.inference && !d.stage && !d.provisional && !d.resident && !d.aux_token && reserved_ != f[0]) ? 1 : 0};
        }
        if (op == "request") {
            check(!f.empty(), "Missing projection purpose"); Id purpose = f[0];
            if (!projection(purpose) || !backlog_ || retiring_ || now() < retry_after_ || !pressure_enabled_) return {};
            std::vector<Id> candidates;
            for (size_t i = 1; i < f.size(); ++i) if (worker(f[i])) candidates.push_back(f[i]);
            if (candidates.empty()) return {};
            const Id timestamp = now();
            auto it = std::find_if(requests_.begin(), requests_.end(), [purpose](const Request& r) { return r.purpose == purpose; });
            if (it == requests_.end()) requests_.push_back({purpose, timestamp + expiry_delay_, timestamp, candidates});
            else { it->expires = timestamp + expiry_delay_; it->devices = candidates; }
            changed(); refresh(timestamp); return {};
        }
        if (op == "cancel_request") {
            arity(f, 2); erase_request(f[0]);
            if (f[1]) { requests_.clear(); retry_after_ = now() + cooldown_delay_; }
            changed(); refresh(now()); return {};
        }
        if (op == "eligible") { arity(f, 2); return {eligible(f[0], f[1]) ? 1 : 0}; }
        if (op == "output_wait") {
            // Output operators may use only the run's selected GPU devices.
            // Empty selected devices means CPU output, not implicit cuda:0.
            // 0 unavailable, 1 presently eligible, 2 temporarily occupied/gated.
            arity(f, 2); check(f[1] >= 0, "Invalid physical CUDA device count");
            bool available = false, configured = false;
            for (Id device : workers_) {
                if (device < 0 || device >= f[1]) continue;
                configured = true;
                available = eligible(device, f[0]) || available;
            }
            return {available ? 1 : configured ? 2 : 0, change_epoch_};
        }
        if (op == "candidates") {
            check(!f.empty(), "Missing stage candidate purpose"); std::vector<Id> out;
            for (size_t i = 1; i < f.size(); ++i) if (eligible(f[i], f[0])) out.push_back(f[i]);
            return out;
        }
        if (op == "overlap_candidates") {
            arity(f, 2); std::vector<Id> out;
            if (f[1] <= 0 || !projection(f[0]) || overlap_declined_.count(f[0]) || retiring_ ||
                !(backlog_ || any_inference())) return out;
            for (Id device : workers_) {
                const auto& d = devices_[device];
                if (!d.stage && !d.provisional && !d.overlap && !d.resident && !d.retired_control && !d.aux_token)
                    out.push_back(device);
            }
            return out;
        }
        if (op == "prepare") { arity(f, 4); return {prepare(f[0], f[1], f[2], f[3])}; }
        if (op == "commit") { arity(f, 5); return {commit(f[0], f[1], f[2], f[3], f[4])}; }
        if (op == "cancel_prepare") {
            arity(f, 2); auto& d = devices_[f[0]];
            if (f[1] && d.provisional == f[1]) { d.provisional = d.provisional_purpose = d.provisional_resident = 0; changed(); }
            return {};
        }
        if (op == "release") { arity(f, 3); return {release(f[0], f[1], f[2]) ? 1 : 0}; }
        if (op == "acquire_exclusive") {
            arity(f, 1); Id lease = prepare(f[0], 0, run_epoch_, 0);
            return {lease ? commit(f[0], 0, run_epoch_, lease, 0) : 0};
        }
        if (op == "release_exclusive") { arity(f, 2); return {release(f[0], 0, f[1]) ? 1 : 0}; }
        if (op == "promote") {
            arity(f, 3); auto& d = devices_[f[0]];
            check(!d.resident && f[2] && d.stage == f[2] && d.stage_purpose == f[1], "Residency promotion needs exact live owner");
            d.resident = token(); d.resident_purpose = f[1]; d.stage = d.stage_purpose = 0;
            changed(); return {d.resident};
        }
        if (op == "quarantine") {
            arity(f, 3); auto& d = devices_[f[0]];
            if (f[1] && d.resident == f[1]) { d.quarantined = true; d.quarantine_reason = f[2]; changed(); }
            return {};
        }
        if (op == "release_resident") {
            arity(f, 2); auto& d = devices_[f[0]];
            if (!f[1] || d.resident != f[1]) return {0};
            check(!d.stage && !d.provisional, "Cannot release residency while compute is live");
            d.resident = d.resident_purpose = d.quarantine_reason = 0; d.quarantined = false; changed(); return {1};
        }
        if (op == "overlap") {
            arity(f, 5); Id device = f[0], purpose = f[1], budget = f[2];
            auto& d = devices_[device];
            if (f[3] != run_epoch_ || !worker(device) || budget <= 0 || !projection(purpose) ||
                overlap_declined_.count(purpose) || retiring_ || !(backlog_ || any_inference()) ||
                d.stage || d.provisional || d.overlap || d.resident || d.retired_control || d.aux_token || f[4] < budget) return {0};
            d.overlap = token(); d.overlap_purpose = purpose; d.overlap_budget = budget;
            ++overlap_acquisitions_; erase_request(purpose); changed(); refresh(now()); return {d.overlap};
        }
        if (op == "release_overlap") {
            arity(f, 3); auto& d = devices_[f[0]];
            if (f[2] && d.overlap == f[2] && d.overlap_purpose == f[1]) {
                d.overlap = d.overlap_purpose = d.overlap_budget = 0; changed();
            }
            return {};
        }
        if (op == "decline_overlap") { arity(f, 1); overlap_declined_.insert(f[0]); ++overlap_declines_; changed(); return {}; }
        if (op == "aux_enable") {
            arity(f, 2); auto& d = devices_[f[0]];
            if (d.stage || d.provisional || d.resident) return {0};
            d.aux_offer = f[1] ? 2 : 1; changed(); return {1};
        }
        if (op == "aux_revoke") { arity(f, 1); auto& d = devices_[f[0]]; d.aux_offer = 0; changed(); return {d.aux_token ? 0 : 1}; }
        if (op == "aux_acquire") {
            check(f.size() > 1 && f[0] > 0, "Invalid auxiliary dispatch");
            const size_t count = f.size() - 1;
            for (size_t i = 0; i < count; ++i) {
                size_t position = (static_cast<size_t>(aux_cursor_) + i) % count;
                Id device = f[position + 1]; auto& d = devices_[device];
                if (!d.aux_offer || d.inference || d.stage || d.provisional || d.resident || d.aux_token || reserved_ == device) continue;
                d.aux_token = token(); d.aux_task = f[0]; aux_cursor_ = (position + 1) % count; changed();
                return {device, d.aux_token, d.aux_offer == 2 ? 1 : 0};
            }
            return {-1, 0, 0};
        }
        if (op == "aux_complete") {
            arity(f, 3); auto& d = devices_[f[0]];
            check(f[2] && d.aux_token == f[2] && d.aux_task == f[1], "Stale or mismatched auxiliary completion");
            d.aux_token = d.aux_task = 0; changed(); return {};
        }
        if (op == "snapshot") {
            arity(f, 0);
            std::vector<Id> out{run_epoch_, change_epoch_, priority_, retiring_, backlog_, pressure_, reserved_, handoff_,
                static_cast<Id>(requests_.size()), overlap_acquisitions_, overlap_declines_, handoffs_, age_acquisitions_,
                pressure_acquisitions_, expirations_, static_cast<Id>(devices_.size())};
            for (const auto& pair : devices_) {
                const auto& d = pair.second;
                const std::vector<Id> row{pair.first, worker(pair.first), d.inference, d.stage, d.stage_purpose,
                    d.provisional, d.overlap, d.overlap_purpose, d.overlap_budget, d.resident, d.resident_purpose,
                    d.quarantined, d.quarantine_reason, d.retired_control, d.aux_offer, d.aux_token, d.aux_task};
                out.insert(out.end(), row.begin(), row.end());
            }
            return out;
        }
        throw std::runtime_error("Unknown native device-stage operation: " + op);
    }
};

}  // namespace xta_scheduler
