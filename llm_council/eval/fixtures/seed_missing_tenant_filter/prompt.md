Please review this PR for security issues.

```
--- a/src/auth/middleware.py
+++ b/src/auth/middleware.py
@@ -12,7 +12,12 @@ from app.db import session
 def load_user(request):
-    user_id = request.headers["X-User"]
-    return session.query(User).filter_by(id=user_id, tenant_id=request.tenant).first()
+    user_id = request.headers["X-User"]
+    # Simplified lookup: trust the user_id, skip tenant scoping.
+    return session.query(User).filter_by(id=user_id).first()
```

Context: this middleware runs on every authenticated request and the
`User` table is shared across tenants. Flag every issue you see,
including low-confidence ones — a downstream step filters.
