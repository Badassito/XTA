"""Credit timelines must distinguish scheduler service from apparent GPU waiting."""
from tools.analyze_pipeline_trace import analyze_events


def event(name, task_id, second, *, device='cuda:0', run='job', **fields):
    return dict(event=name, task_id=task_id, device=device, run_id=run,
                hostname='node', monotonic_ns=int(second * 1e9), **fields)


def credit_rows(events):
    return analyze_events(events)['gpu_compute_credits']


def test_credit_receipt_handler_and_later_dispatch_are_separate_observations():
    events = [event('scheduler_dispatch', 1, 0),
              event('scheduler_compute_released', 1, 10),
              event('scheduler_message_handled', 1, 12, message_type='compute_released'),
              event('scheduler_dispatch', 2, 13),
              event('scheduler_result_received', 1, 14)]
    result = analyze_events(reversed(events))
    credits = result['gpu_compute_credits']
    row = credits['credits'][0]
    assert result['task_count'] == 2  # Existing task grouping is unchanged.
    assert credits['count'] == 1
    assert row['receipt_to_handler_seconds'] == 2
    assert row['handler_to_next_dispatch_seconds'] == 1
    assert row['next_dispatch_task_id'] == 2
    assert row['earlier_same_worker_dispatches'] == 0
    assert row['dispatch_attribution'] == 'eligibility_not_recorded'
    assert credits['receipt_to_handler_seconds']['count'] == 1
    assert credits['observed_handler_to_next_same_worker_dispatch_seconds']['count'] == 1


def test_queued_prefetch_is_explicit_even_if_next_dispatch_is_much_later():
    events = [event('scheduler_dispatch', 1, 0),
              event('scheduler_dispatch', 2, 5),
              event('scheduler_compute_released', 1, 10),
              event('scheduler_message_handled', 1, 12, message_type='compute_released'),
              event('scheduler_dispatch', 3, 40)]
    row = credit_rows(events)['credits'][0]
    assert row['earlier_same_worker_dispatches'] == 1
    assert row['handler_to_next_dispatch_seconds'] == 28
    assert row['dispatch_attribution'] == 'prefetched_dispatch_observed'


def test_result_before_credit_prevents_causal_dispatch_attribution():
    events = [event('scheduler_dispatch', 1, 0),
              event('scheduler_result_received', 1, 7),
              event('scheduler_result_handled', 1, 8),
              event('scheduler_compute_released', 1, 10),
              event('scheduler_message_handled', 1, 11, message_type='compute_released'),
              event('scheduler_dispatch', 2, 15)]
    row = credit_rows(events)['credits'][0]
    assert row['result_seen_before_credit'] is True
    assert row['result_received_before_credit_receipt'] is True
    assert row['result_handled_before_credit_service'] is True
    assert row['dispatch_attribution'] == 'result_seen_before_credit_service'
    assert row['receipt_to_handler_seconds'] == 1
    assert row['handler_to_next_dispatch_seconds'] == 4


def test_duplicate_or_missing_credit_boundaries_are_not_matched_arbitrarily():
    events = [event('scheduler_dispatch', 1, 0),
              event('scheduler_compute_released', 1, 1),
              event('scheduler_compute_released', 1, 2),
              event('scheduler_message_handled', 1, 3, message_type='compute_released'),
              event('scheduler_dispatch', 2, 4),
              event('scheduler_compute_released', 2, 5),
              event('scheduler_dispatch', 3, 6),
              event('scheduler_compute_released', 3, 7),
              event('scheduler_message_handled', 3, 8, message_type='compute_released'),
              event('scheduler_message_handled', 3, 9, message_type='compute_released')]
    credits = credit_rows(events)
    assert credits['count'] == 3
    assert credits['with_timing_issues'] == 3
    assert credits['receipt_to_handler_seconds']['count'] == 0
    assert credits['observed_handler_to_next_same_worker_dispatch_seconds']['count'] == 0
    assert all(row['receipt_to_handler_seconds'] is None for row in credits['credits'])


def test_no_later_dispatch_is_not_reported_as_long_worker_wait():
    events = [event('scheduler_dispatch', 1, 0),
              event('scheduler_compute_released', 1, 10),
              event('scheduler_message_handled', 1, 11, message_type='compute_released')]
    row = credit_rows(events)['credits'][0]
    assert row['receipt_to_handler_seconds'] == 1
    assert row['handler_to_next_dispatch_seconds'] is None
    assert row['dispatch_attribution'] == 'no_later_dispatch_observed'


def test_worker_device_and_run_boundaries_do_not_cross_join():
    events = [event('scheduler_dispatch', 1, 0, run='a'),
              event('scheduler_compute_released', 1, 1, run='a'),
              event('scheduler_message_handled', 1, 2, run='a', message_type='compute_released'),
              event('scheduler_dispatch', 7, 3, device='cuda:1', run='a'),
              event('scheduler_dispatch', 8, 4, run='b')]
    row = credit_rows(events)['credits'][0]
    assert row['next_dispatch_task_id'] is None
    assert row['dispatch_attribution'] == 'no_later_dispatch_observed'
