
_ACTION_TEXT = {
    "workspace.created": "created the workspace",
    "workspace.updated": "updated the workspace",
    "workspace.deleted": "deleted the workspace",
    "member.joined": "joined the workspace",
    "member.role_changed": "changed a member's role",
    "member.removed": "removed a member",
    "invite.created": "generated the share link",
    "invite.generated": "generated the share link",
    "invite.regenerated": "regenerated the share link",
    "invite.revoked": "revoked the share link",
    "join_request.created": "requested access",
    "join_request.approved": "approved an access request",
    "join_request.rejected": "denied an access request",
    "project.created": "created a project",
    "project.updated": "updated a project",
    "project.deleted": "deleted a project",
    "file.uploaded": "uploaded a file",
    "file.deleted": "deleted a file",
}


def describe_activity(action: str, metadata: dict | None = None) -> str:
    meta = metadata or {}
    if action == "workspace.updated":
        parts = []
        if meta.get("visibility"):
            parts.append(f"set the workspace to {meta['visibility']}")
        if meta.get("name"):
            parts.append(f'renamed the workspace to "{meta["name"]}"')
        if parts:
            return " and ".join(parts)
    if action in _ACTION_TEXT:
        return _ACTION_TEXT[action]
    resource, _, verb = action.partition(".")  
    if verb:
        return f"{verb.replace('_', ' ')} ({resource.replace('_', ' ')})"
    return action.replace("_", " ")
