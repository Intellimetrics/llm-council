# Product vocabulary

This registry is the source of truth for operator-facing CLI, MCP, setup, and
documentation labels. A participant becomes a peer only after runtime routing
selects it for a council run.

| Concept | Description | Allowed UI label | Disallowed or confusable labels |
| --- | --- | --- | --- |
| Council | A read-only consultation that gathers independent responses and reports their combined result. | Council | Panel, workflow, swarm |
| Setup Preset | A setup-time bundle that determines which participant routes and runtime modes are written to project config. | Setup Preset | Runtime Mode, route, profile |
| Runtime Mode | A named runtime routing configuration selected by `--mode` or the project default. | Runtime Mode | Setup Preset, preset, workflow |
| Participant | A configured CLI or model endpoint that is available for runtime selection. | Participant | Peer, reviewer, seat |
| Peer | A participant selected to respond, deliberate, or vote in one council run. | Peer | Participant when describing config, reviewer, seat |
| Host | The primary developer agent that invokes the council and decides what to do with its advice. | Host | Current agent, orchestrator, reviewer |
| Synthesis Chair | The optional participant that summarizes peer responses without replacing their votes. | Synthesis Chair | Judge, host, lead reviewer |
| Transcript | The Markdown, JSON, or HTML record of one council run. | Transcript, HTML Transcript | Report, log, dashboard |
| Partial Result | Completed findings preserved when the request deadline stops other peer work; quorum may still be met. | Partial result | Complete review, degraded as a synonym |

Technical compatibility names such as the `current` input, `participants` API
field, `other_cli_peers` strategy, and deprecated CLI aliases remain unchanged.
They do not create additional UI labels.
