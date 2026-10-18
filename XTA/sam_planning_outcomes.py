"""Compact requested-work diagnostics shared by SAM execution and reports."""


def sam_planning_omission(stats, *, operation, scope_id='', view_name=''):
    status = str(stats.get('sam_planning_status', stats.get('planning_status', '')))
    failed = {'unresolved', 'incomplete', 'invalid'}
    groups = stats.get('sam_group_planning_receipts', stats.get('group_planning_receipts', ())) or ()
    unresolved = [group for group in groups if str(group.get('status')) in failed]
    unresolved_count = max(len(unresolved), int(stats.get('sam_unresolved_groups', 0)))
    if status not in failed | {'partial'} and not unresolved_count:
        return None
    reasons = set(stats.get('sam_planning_reasons', stats.get('planning_reasons', ())) or ())
    group_reasons = {}
    for group in unresolved:
        for reason in set(group.get('reasons') or ()):
            reason = str(reason)
            group_reasons[reason] = group_reasons.get(reason, 0) + 1
            reasons.add(reason)
    if unresolved_count and not unresolved:
        group_reasons = {str(reason): int(count)
            for reason, count in (stats.get('sam_group_refusal_reasons') or {}).items()}
        reasons.update(group_reasons)
    return dict(operation=str(operation), scope_id=str(scope_id or stats.get('scope_id', '')),
        view=str(view_name or stats.get('view', stats.get('view_name', ''))),
        pass_index=int(stats.get('pass_index', 1)), status=status,
        reasons=sorted(map(str, reasons)), unresolved_plan=status in failed,
        unresolved_group_count=unresolved_count, group_refusal_reasons=dict(sorted(group_reasons.items())))
