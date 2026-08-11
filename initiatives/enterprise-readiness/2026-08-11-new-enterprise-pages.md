# Enterprise Redesign — New Pages for Multi-Tenancy

**Date:** 2026-08-11
**Status:** Design proposal (subagent-produced, text-based design brief).
Companion to
[2026-08-11-redesign-existing-pages.md](2026-08-11-redesign-existing-pages.md)
and depends on the backend work described in
[2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md) §1
(auth), §2 (multi-tenancy), and §5 (audit logging) actually being built —
**this document designs the frontend/UX only; no backend implementation
is proposed here.**

## Correction before reading further: the audit log page already exists

This design was originally produced assuming an Audit Log viewer needed to
be built from scratch. **A separate, independent frontend audit found this
is incorrect: `pages/AuditLogPage.tsx` already exists in the routed app**
(`src/App.tsx:31-92` route list). The §4 design below should therefore be
treated as **a redesign/extension brief for the existing page** (add
compliance-grade filtering, export, and the richer event schema once
backend-blockers §5's real audit-event table lands) rather than a
from-scratch build. The rest of this document's proposed pages (Login/SSO,
Org/Workspace Settings, RBAC, Usage/Billing, API Keys) were independently
confirmed to have **no existing equivalent** in the current route list —
those five remain genuinely new.

## Current frontend fit (for all new pages)

Skyulf uses a left navigation shell (`components/Layout.tsx`), Tailwind
light/dark theming, React Router routes, and existing page-level loading/
error/empty state components. Reuse across every new page below:
`Button`, `Input`, `FormField`, `Badge`, `Tooltip`, `ModalShell`,
`EmptyState`, `LoadingState`, `ErrorState`, `StatusBadge`, Recharts +
`useChartTheme()`.

New shared primitives needed across multiple pages below: `DataTable`,
`RoleBadge`, `StatCard`, `QuotaMeter`, `SettingsNav`, `FilterBar`,
`PermissionMatrix`. **These overlap significantly with the `DataTable`/
`StatusBadge` consolidation already recommended in the existing-pages
redesign doc — build them once, shared across both efforts, not twice.**

---

## 1. Login / SSO — genuinely new

**Route:** `/login`, outside the authenticated app shell.

**Layout.** Centered auth card: Skyulf mark, "Sign in to your workspace,"
email-first form, support/contact + privacy/terms footer.

**Flow.**
1. User enters email, selects Continue.
2. Domain discovery resolves to: org SSO configured → show "Continue with
   *Acme*" (OIDC/SAML redirect); password enabled → reveal password field;
   both → SSO primary, "Use password instead" as deliberate fallback.
3. Explicit "Continue with your organization" opens a compact form
   accepting org slug or work email.
4. Redirect target retained through auth, validated server-side against
   workspace access.

**States.** Loading ("Finding your organization…"); invalid credentials/
SSO failure (non-enumerating error copy + retry); workspace-not-found
("Request access" CTA — name/email/org/note, success confirmation without
revealing tenant existence); no unrestricted self-serve signup — "Contact
sales" instead, consistent with the AGPL+commercial licensing model noted
in backend-blockers §10.

**Components.** Reuse `FormField`, `Input`, `Button`, `ModalShell`,
`ErrorState`. New: `AuthCard`, `PasswordField`, `SsoProviderButton`.

## 2. Organization & Workspace Settings — genuinely new

**Routes:** `/settings/organization`, `/settings/workspace`.

**Layout.** Authenticated shell + settings header: breadcrumb ("Settings /
Organization"), title/description, contextual admin actions, persistent
settings left-nav (desktop) / scrollable tabs (mobile).

**Organization page.** Profile card (name, logo, slug, verified domains,
primary contact); workspace directory (name, description, member count,
storage/compute summary, "Switch workspace"); member summary (count,
pending invites, link to Members & Roles).

**Workspace page.** Name/description/retention policy/region display;
**default compute limits** (max concurrent jobs, default CPU/GPU class,
per-job duration — display server-enforced entitlement values, do not
imply client-side enforcement); **storage quota** via `QuotaMeter`; danger
zone for destructive actions, Owner-gated.

**Invite flow.** Modal: multi-email input, workspace selector, role
selector (Admin/Editor/Viewer), optional message, review/send. Pending
Invitations table: email, role, workspace, inviter, sent/expires,
resend/revoke.

**States.** New-org empty state = profile setup checklist + prominent
invite CTA; per-card loading skeletons; scoped `ErrorState` per card so one
failed section doesn't block the rest.

## 3. Member & Role Management (RBAC) — genuinely new

**Route:** `/settings/members`.

**Layout.** Header ("Members & roles," workspace context, member count,
"Invite members" button) → member table → permission-matrix card.

**Member table.** Member, email, workspace role (`RoleBadge`), status, last
active, actions (change role / remove access / resend-revoke invite).

**Role model** (workspace-scoped; org directory can list a person across
multiple workspaces, each with its own role):

| Capability | Owner | Admin | Editor | Viewer |
|---|---:|---:|---:|---:|
| View pipelines/data/results | Yes | Yes | Yes | Yes |
| Create/edit pipelines | Yes | Yes | Yes | No |
| Run/cancel jobs | Yes | Yes | Yes | No |
| Deploy models | Yes | Yes | No | No |
| Manage members | Yes | Yes | No | No |
| Manage billing/licensing | Yes | No | No | No |

**Rules:** Owners can't remove/demote the last Owner; a user can't grant a
role above their own; removals require confirmation naming the affected
workspace; role changes only show success after backend authorization
actually succeeds (no optimistic-before-confirmed UI).

**Permission matrix.** Ship as a **fixed, non-editable** matrix for the
first release rather than a full policy editor — simpler, and can be
replaced later with custom roles without changing the table's interaction
model.

**States.** Empty (only current Owner) explains roles + invite CTA;
independent loading/error for table vs matrix; populated view is
sortable/filterable by role/status.

## 4. Compliance Audit Log Viewer — **redesign of existing `AuditLogPage.tsx`**, not new

**Route:** `/settings/audit-log` (or keep the existing route if
`AuditLogPage.tsx` is already mounted elsewhere — confirm during
implementation planning rather than assuming this design's proposed path).
Restrict to Owner/Admin or a future audit-reader entitlement — **this
access restriction does not exist today**, since there's no real auth yet
(backend-blockers §1); note this as a hard dependency, not an
implementation detail.

**Layout.** Header ("Audit log," retention note, Export CSV); filter bar
(full-text search, actor, action, resource type, outcome, workspace,
date-range — default last 30 days); paginated/cursor-based table with
sticky header.

**Table columns.** Timestamp (localized + accessible ISO value), Actor
(user or service account), Action, Resource (type/name/immutable ID),
Workspace, Outcome (Success/Denied/Failed), Details expander (correlation
ID, source IP if retained, actor type, before/after metadata where safe —
**never** render secrets/token material/sensitive payloads).

Search/filter state reflected in URL query params so compliance officers
can share a filtered view; large exports become an async download
notification rather than blocking the browser.

**States.** New-org: "No auditable activity yet" + explanation that events
appear once enterprise audit logging (backend-blockers §5) is enabled;
loading = row skeletons; error preserves prior results + retry; populated =
result count, pagination, filter chips, CSV download.

**Implementation note:** since this page already exists, the actual next
step is auditing `pages/AuditLogPage.tsx`'s current capability against this
brief (what filters/columns/export it already has vs what's missing) —
this should be a fast follow-up investigation, not a from-scratch build
estimate.

## 5. Usage, Billing & Quota Dashboard — genuinely new

**Route:** `/settings/usage`.

**Layout.** Header: org, active plan/license status, billing period, one
contextual CTA — managed tenant → "Upgrade plan"/"Contact sales";
self-managed AGPL tenant → "Explore commercial licensing"/"Contact sales";
commercial self-managed → support/license contact. **Entitlement/quota
enforcement must remain server-side** — no frontend-only feature gates.

**Main content.** Four `StatCard`s (storage, compute-hours, active jobs,
dataset count — each vs quota) with `QuotaMeter`, percentage, reset date,
warning/error thresholds (e.g. 80%/100%). Usage-over-time chart (Recharts +
`useChartTheme()`); breakdown controls + period selector; usage detail
table or downloadable report; "Plan limits" side card with upgrade route.

**States.** New org = zero-use cards + "Create dataset"/"Run first job"
links; independent card/chart loading skeletons; independent retry per
section; populated = real quota status with threshold warning banner.

## 6. API Keys / Service Accounts — genuinely new

**Route:** `/settings/api-access`, Owner/Admin only.

**Layout.** Header + "Create service account" button; two sections:
Service Accounts, API Keys.

**Service accounts table.** Name, description, workspace scope, assigned
scopes, created by/date, last used, status, actions. Keys belong to a
service account (not directly to a human) where possible.

**API-key creation wizard.** Name + service account → workspace scope →
permission scopes (mirroring the RBAC vocabulary: pipelines read/write,
datasets read/write, jobs read/run/cancel, models/deployments read/deploy,
audit/usage read) → optional expiration → review/create. **Raw secret
shown exactly once** at creation with copy control + explicit warning;
list view thereafter shows only prefix/status/creator/created/last-used/
expiration/scopes. Revocation requires confirmation; revoked/expired keys
remain visible as immutable history.

**States.** Empty explains programmatic access + create CTA; independent
table loading/error; populated = sortable/filterable with last-used/expiry
health indicators.

---

## App Shell Integration

Today, most pages use `Layout.tsx`'s left rail, while Canvas adds its own
internal navbar (same fragmentation issue flagged independently in the
existing-pages redesign doc's cross-page section). Consolidate around one
authenticated shell:

1. **Global top bar:** logo, organization/workspace switcher ("Acme /
   Fraud Detection"), current-workspace role indicator, search/command
   palette entry, notifications, user avatar menu (profile/settings/sign
   out).
2. **Left navigation**, keeping product work primary (Dashboard, Jobs, EDA,
   ML Canvas, Data Sources, Registry, Deployments) — grouped per the
   existing-pages redesign doc's **Build/Operate/Observe** scheme, plus a
   new **Settings** group for everything in this document.
3. **Settings entry:** bottom-of-rail gear → `/settings/organization`,
   which supplies Organization, Workspace, Members & Roles, Audit Log,
   Usage & Billing, API Access as a sub-nav.
4. **Canvas continuity:** Canvas keeps its palette/toolbar/properties
   panel inside the shared shell; the global top bar supplies identity/
   workspace context everywhere; Canvas's internal view tabs become
   content-level controls, not a competing app header.

Use the existing slate/indigo/purple visual language, dark-mode tokens,
card borders, and shared state components so the settings/admin area feels
like the same product, not a bolted-on console.

## Dependencies & Sequencing

This entire document's pages are **UX designs for backend capability that
doesn't exist yet.** None of pages 1-3, 5, or 6 can be implemented for real
before backend-blockers §1 (auth) and §2 (multi-tenancy) land — they can be
built as static/mocked UI in parallel with backend work, but should not be
considered "done" until wired to real endpoints. Page 4 (audit log) is the
one exception — it already has a real frontend page and can be
incrementally extended as backend-blockers §5's real audit-event table
comes online, without waiting for §1/§2.
