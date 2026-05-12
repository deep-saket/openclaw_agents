const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatLog = document.getElementById("chatLog");
const executionChart = document.getElementById("executionChart");
const executionNarrative = document.getElementById("executionNarrative");
const thinkingTrace = document.getElementById("thinkingTrace");
const planTreeView = document.getElementById("planTreeView");
const planPrevButton = document.getElementById("planPrevButton");
const planNextButton = document.getElementById("planNextButton");
const planSnapshotLabel = document.getElementById("planSnapshotLabel");
const snapshotPicker = document.getElementById("snapshotPicker");
const stateJson = document.getElementById("stateJson");
const memoryJson = document.getElementById("memoryJson");
const nodesExplored = document.getElementById("nodesExplored");
const nodePromptJson = document.getElementById("nodePromptJson");
const nodeOutputJson = document.getElementById("nodeOutputJson");
const nodeDiffJson = document.getElementById("nodeDiffJson");
const collapseButton = document.getElementById("collapseButton");
const appShell = document.getElementById("appShell");
const sessionIdInput = document.getElementById("sessionId");
const demoUsersEl = document.getElementById("demoUsers");
const llmStatusEl = document.getElementById("llmStatus");
const callPulseEl = document.getElementById("callPulse");
const callStatusTextEl = document.getElementById("callStatusText");
const callOpenLinkEl = document.getElementById("callOpenLink");
const callStopButton = document.getElementById("callStopButton");
const callTranscriptEl = document.getElementById("callTranscript");

const viewState = {
  snapshots: [],
  nodeEntries: [],
  selectedNodeId: null,
  liveHops: [],
  collapsed: false,
  demoUsers: [],
  planSnapshots: [],
  planSnapshotIndex: 0,
  selectedPlanNodeId: null,
  transcriptPoller: null,
  liveStatePoller: null,
  callSessionId: "",
  callActive: false,
};

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderExecutionChart(hops) {
  executionChart.innerHTML = "";
  if (!hops.length) {
    executionChart.textContent = "No execution hops yet.";
    return;
  }

  for (const hop of hops) {
    const row = document.createElement("div");
    row.className = "hop-row";

    const nodeHistory = Array.isArray(hop.node_history) ? hop.node_history.join(" -> ") : "";
    const toolCalls = (hop.events || [])
      .filter((event) => event.event === "tool_call")
      .map((event) => `${event.tool_name} [${event.status}]`)
      .join(" | ");

    row.innerHTML = `
      <div class="hop-title">Hop ${hop.hop}: target=${hop.response_target} | elapsed=${formatMs(hop.elapsed_ms)}</div>
      <div class="hop-nodes">Nodes: ${nodeHistory || "(none)"}</div>
      <div class="hop-tools">Tools: ${toolCalls || "(none)"}</div>
    `;
    executionChart.appendChild(row);
  }
}

function formatMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
}

function summarizePlan(planProposal) {
  if (!planProposal || typeof planProposal !== "object") return "(no plan)";
  const outline = String(planProposal.plan_outline || "").trim();
  if (outline) return outline;
  const actions = Array.isArray(planProposal.next_actions) ? planProposal.next_actions : [];
  if (actions.length) return `next actions: ${actions.join(", ")}`;
  const intent = String(planProposal.intent || "").trim();
  return intent ? `intent: ${intent}` : "(no plan)";
}

function summarizeToolReason(state) {
  const decision = state && typeof state === "object" ? state.decision : null;
  if (!decision || typeof decision !== "object") return null;
  const thought = String(decision.thought || "").trim();
  const toolCall = decision.tool_call && typeof decision.tool_call === "object" ? decision.tool_call : null;
  if (!toolCall) return thought || null;
  const toolName = String(toolCall.tool_name || "").trim();
  const args = toolCall.arguments ? JSON.stringify(toolCall.arguments) : "{}";
  if (thought) return `Executing ${toolName} because: ${thought} | args=${args}`;
  return `Executing ${toolName} | args=${args}`;
}

function renderExecutionNarrative(hops) {
  executionNarrative.innerHTML = "";
  if (!Array.isArray(hops) || !hops.length) {
    executionNarrative.textContent = "No narrative yet.";
    return;
  }

  const lines = [];
  for (const hop of hops) {
    const state = hop.state || {};
    const nodeHistory = Array.isArray(hop.node_history) ? hop.node_history.join(" -> ") : "(none)";
    const planSummary = summarizePlan(state.plan_proposal);
    const toolReason = summarizeToolReason(state);
    const tools = (hop.events || [])
      .filter((event) => event.event === "tool_call")
      .map((event) => `${event.tool_name} [${event.status}] in ${formatMs(event.duration_ms)}`);
    const nodeDurations = (hop.events || [])
      .filter((event) => event.event === "node_finished")
      .map((event) => `${event.node_name}: ${formatMs(event.duration_ms)}`);
    const phase = String(hop.conversation_phase || "unknown");
    const nodeList = Array.isArray(hop.node_history) ? hop.node_history : [];
    const planState = state && typeof state === "object" ? state.conversation_plan : null;
    const planCurrent = planState && typeof planState === "object" ? String(planState.current_node_id || "") : "";
    const planVersion = planState && typeof planState === "object" ? Number(planState.version || 1) : null;

    lines.push(`Hop ${hop.hop} started.`);
    lines.push(`Path: ${nodeHistory}`);
    lines.push(`Phase: ${phase}`);
    if (nodeList.includes("plan_proposal")) lines.push("Constructing plan proposal for this turn.");
    if (nodeList.includes("tool_execution")) lines.push("Executing tool calls selected by planner.");
    if (nodeList.includes("relevant_response") || nodeList.includes("irrelevant_response")) {
      lines.push("Constructing final response package.");
    }
    if (planCurrent) {
      lines.push(`Plan tree: v${planVersion} at node=${planCurrent}`);
    }
    lines.push(`Plan: ${planSummary}`);
    if (toolReason) lines.push(toolReason);
    if (tools.length) lines.push(`Tool results: ${tools.join(" | ")}`);
    if (nodeDurations.length) lines.push(`Node timings: ${nodeDurations.join(" | ")}`);
    lines.push(`Response packaged for target=${hop.response_target} in ${formatMs(hop.elapsed_ms)}.`);
    lines.push(`Output: ${String(hop.response || "").trim() || "(empty response)"}`);
    lines.push("");
  }

  for (const line of lines) {
    const div = document.createElement("div");
    div.className = "narrative-line";
    div.textContent = line;
    executionNarrative.appendChild(div);
  }
  executionNarrative.scrollTop = executionNarrative.scrollHeight;
}

function renderThinking(hops) {
  thinkingTrace.innerHTML = "";
  const lines = [];
  for (const hop of hops) {
    lines.push(`--- hop ${hop.hop} ---`);
    for (const line of hop.thinking || []) {
      lines.push(line);
    }
  }

  if (!lines.length) {
    thinkingTrace.textContent = "No trace yet.";
    return;
  }

  for (const line of lines) {
    const p = document.createElement("div");
    p.className = "trace-line";
    p.textContent = line;
    thinkingTrace.appendChild(p);
  }
  thinkingTrace.scrollTop = thinkingTrace.scrollHeight;
}

function resolveConversationPlan(state) {
  if (!state || typeof state !== "object") return null;
  if (state.conversation_plan && typeof state.conversation_plan === "object") {
    return state.conversation_plan;
  }
  if (state.plan_proposal && typeof state.plan_proposal === "object") {
    const embedded = state.plan_proposal.conversation_plan;
    if (embedded && typeof embedded === "object") return embedded;
  }
  return null;
}

function extractPlanSnapshots(hops, finalState) {
  const snapshots = [];
  for (const hop of hops || []) {
    const state = hop && typeof hop === "object" ? hop.state || {} : {};
    const plan = resolveConversationPlan(state);
    if (!plan || typeof plan !== "object") continue;
    snapshots.push({
      id: `hop-${hop.hop}`,
      label: `Hop ${hop.hop}`,
      plan,
      response_target: String(hop.response_target || ""),
    });
  }
  const finalPlan = resolveConversationPlan(finalState);
  if (finalPlan && typeof finalPlan === "object") {
    snapshots.push({
      id: "final",
      label: "Final",
      plan: finalPlan,
      response_target: String(finalState.response_target || ""),
    });
  }
  return snapshots;
}

function normalizeStatus(value) {
  const status = String(value || "pending").trim().toLowerCase();
  if (["in_progress", "pending", "done", "skipped", "blocked"].includes(status)) return status;
  return "pending";
}

function statusColors(status, isCurrent = false) {
  const normalized = normalizeStatus(status);
  const palette = {
    in_progress: { fill: "#e6f0ff", stroke: "#0d66d0" },
    pending: { fill: "#fff6e8", stroke: "#be7c00" },
    done: { fill: "#e7f8ef", stroke: "#0d8a52" },
    skipped: { fill: "#f3f4f7", stroke: "#6b7280" },
    blocked: { fill: "#ffe8e8", stroke: "#bf2d2d" },
  };
  const base = palette[normalized] || palette.pending;
  if (!isCurrent) return base;
  return { fill: base.fill, stroke: "#084d9a" };
}

function nodeStatusClass(status) {
  const normalized = normalizeStatus(status);
  return normalized.replace("_", "-");
}

function wrapLabel(text, maxChars = 28) {
  const source = String(text || "").trim();
  if (!source) return [""];
  const words = source.split(/\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxChars) {
      current = candidate;
      continue;
    }
    if (current) lines.push(current);
    current = word;
  }
  if (current) lines.push(current);
  return lines.slice(0, 2);
}

function buildPlanGraphData(plan) {
  if (!plan || typeof plan !== "object") return null;
  const nodes = Array.isArray(plan.nodes) ? plan.nodes.filter((node) => node && typeof node === "object") : [];
  if (!nodes.length) return null;

  const nodeById = new Map();
  const children = new Map();
  const incoming = new Map();
  for (const node of nodes) {
    const nodeId = String(node.id || "").trim();
    if (!nodeId) continue;
    nodeById.set(nodeId, node);
    children.set(nodeId, []);
    incoming.set(nodeId, 0);
  }

  const edgesRaw = Array.isArray(plan.edges) ? plan.edges.filter((edge) => edge && typeof edge === "object") : [];
  const edges = [];
  for (const edge of edgesRaw) {
    const from = String(edge.from || "").trim();
    const to = String(edge.to || "").trim();
    const condition = String(edge.condition || "").trim();
    if (!from || !to || !nodeById.has(from) || !nodeById.has(to)) continue;
    children.get(from).push({ to, condition });
    incoming.set(to, (incoming.get(to) || 0) + 1);
    edges.push({ from, to, condition });
  }

  let root = String(plan.root_node_id || "").trim();
  if (!root || !nodeById.has(root)) {
    root = [...incoming.entries()].find((entry) => entry[1] === 0)?.[0] || String(nodes[0].id || "");
  }

  const levels = new Map();
  const visited = new Set();
  const queue = [{ nodeId: root, depth: 0 }];
  while (queue.length) {
    const current = queue.shift();
    if (!current) break;
    const { nodeId, depth } = current;
    if (!nodeById.has(nodeId) || visited.has(nodeId)) continue;
    visited.add(nodeId);
    if (!levels.has(depth)) levels.set(depth, []);
    levels.get(depth).push(nodeId);
    const kids = children.get(nodeId) || [];
    for (const child of kids) queue.push({ nodeId: child.to, depth: depth + 1 });
  }

  const remaining = [...nodeById.keys()].filter((nodeId) => !visited.has(nodeId));
  if (remaining.length) {
    const nextDepth = levels.size ? Math.max(...levels.keys()) + 1 : 0;
    levels.set(nextDepth, remaining);
  }

  return {
    root,
    edges,
    nodeById,
    levels,
    currentNodeId: String(plan.current_node_id || "").trim(),
  };
}

function renderPlanTreeGraph(plan) {
  const graph = buildPlanGraphData(plan);
  if (!graph) {
    const empty = document.createElement("div");
    empty.className = "plan-tree-empty";
    empty.textContent = "No plan graph available for this snapshot.";
    planTreeView.appendChild(empty);
    return;
  }

  const { nodeById, levels, edges, currentNodeId } = graph;
  const nodeWidth = 230;
  const nodeHeight = 64;
  const levelGap = 285;
  const rowGap = 94;
  const marginX = 32;
  const marginY = 24;
  const positions = new Map();
  let maxRows = 1;

  const depthKeys = [...levels.keys()].sort((a, b) => a - b);
  for (const depth of depthKeys) {
    const ids = levels.get(depth) || [];
    maxRows = Math.max(maxRows, ids.length || 1);
    ids.forEach((nodeId, idx) => {
      const x = marginX + depth * levelGap;
      const y = marginY + idx * rowGap;
      positions.set(nodeId, { x, y });
    });
  }

  const width = Math.max(760, marginX * 2 + depthKeys.length * levelGap + nodeWidth);
  const height = Math.max(240, marginY * 2 + maxRows * rowGap + nodeHeight);
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.classList.add("plan-tree-canvas");

  const defs = document.createElementNS(svgNs, "defs");
  const marker = document.createElementNS(svgNs, "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "7");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "3.5");
  marker.setAttribute("orient", "auto");
  const arrow = document.createElementNS(svgNs, "path");
  arrow.setAttribute("d", "M0,0 L10,3.5 L0,7 z");
  arrow.setAttribute("fill", "#9bb0c8");
  marker.appendChild(arrow);
  defs.appendChild(marker);
  svg.appendChild(defs);

  for (const edge of edges) {
    const fromPos = positions.get(edge.from);
    const toPos = positions.get(edge.to);
    if (!fromPos || !toPos) continue;
    const x1 = fromPos.x + nodeWidth;
    const y1 = fromPos.y + nodeHeight / 2;
    const x2 = toPos.x;
    const y2 = toPos.y + nodeHeight / 2;
    const midX = x1 + (x2 - x1) * 0.45;
    const path = document.createElementNS(svgNs, "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`);
    path.setAttribute("class", "plan-tree-edge");
    path.setAttribute("marker-end", "url(#arrowhead)");
    svg.appendChild(path);

    if (edge.condition) {
      const text = document.createElementNS(svgNs, "text");
      text.setAttribute("x", String((x1 + x2) / 2));
      text.setAttribute("y", String((y1 + y2) / 2 - 4));
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("class", "plan-tree-edge-label");
      text.textContent = edge.condition;
      svg.appendChild(text);
    }
  }

  const selectedNodeId = viewState.selectedPlanNodeId || currentNodeId || "";
  for (const [nodeId, position] of positions.entries()) {
    const node = nodeById.get(nodeId);
    if (!node) continue;
    const status = normalizeStatus(node.status);
    const isCurrent = nodeId === currentNodeId;
    const isSelected = nodeId === selectedNodeId;
    const colors = statusColors(status, isCurrent || isSelected);

    const group = document.createElementNS(svgNs, "g");
    group.setAttribute(
      "class",
      `plan-tree-node ${isCurrent ? "current" : ""} ${isSelected ? "selected" : ""}`,
    );
    group.setAttribute("transform", `translate(${position.x}, ${position.y})`);

    const rect = document.createElementNS(svgNs, "rect");
    rect.setAttribute("width", String(nodeWidth));
    rect.setAttribute("height", String(nodeHeight));
    rect.setAttribute("rx", "10");
    rect.setAttribute("fill", colors.fill);
    rect.setAttribute("stroke", colors.stroke);
    group.appendChild(rect);

    const labelLines = wrapLabel(node.label || nodeId);
    labelLines.forEach((line, idx) => {
      const text = document.createElementNS(svgNs, "text");
      text.setAttribute("x", "10");
      text.setAttribute("y", String(20 + idx * 14));
      text.textContent = line;
      group.appendChild(text);
    });

    const meta = document.createElementNS(svgNs, "text");
    meta.setAttribute("x", "10");
    meta.setAttribute("y", "54");
    meta.setAttribute("class", "node-meta");
    meta.textContent = `${status} | ${String(node.owner || "collection_agent")}`;
    group.appendChild(meta);

    group.addEventListener("click", () => {
      viewState.selectedPlanNodeId = nodeId;
      renderPlanTree();
    });
    svg.appendChild(group);
  }

  planTreeView.appendChild(svg);
}

function renderPlanTree() {
  if (!planTreeView || !planSnapshotLabel || !planPrevButton || !planNextButton) {
    return;
  }
  const snapshots = viewState.planSnapshots || [];
  if (!snapshots.length) {
    planSnapshotLabel.textContent = "No plan snapshots";
    planTreeView.textContent = "No plan generated yet.";
    viewState.selectedPlanNodeId = null;
    planPrevButton.disabled = true;
    planNextButton.disabled = true;
    return;
  }

  if (viewState.planSnapshotIndex < 0) viewState.planSnapshotIndex = 0;
  if (viewState.planSnapshotIndex >= snapshots.length) viewState.planSnapshotIndex = snapshots.length - 1;

  const selected = snapshots[viewState.planSnapshotIndex];
  const plan = selected.plan || {};
  const version = Number(plan.version || 1);
  const status = String(plan.status || "active");
  const currentNodeId = String(plan.current_node_id || "");
  planSnapshotLabel.textContent =
    `${selected.label} (${viewState.planSnapshotIndex + 1}/${snapshots.length}) | ` +
    `v${version} | status=${status} | current=${currentNodeId || "-"}`;

  planTreeView.innerHTML = "";
  const planNodes = Array.isArray(plan.nodes) ? plan.nodes : [];
  const planNodeIds = new Set(planNodes.map((node) => String((node && node.id) || "")));
  if (!viewState.selectedPlanNodeId || !planNodeIds.has(String(viewState.selectedPlanNodeId))) {
    viewState.selectedPlanNodeId = currentNodeId || null;
  }
  const legend = document.createElement("div");
  legend.className = "plan-tree-legend";
  legend.innerHTML = `
    <span class="legend-item"><span class="legend-dot in-progress"></span>In Progress</span>
    <span class="legend-item"><span class="legend-dot pending"></span>Pending</span>
    <span class="legend-item"><span class="legend-dot done"></span>Done</span>
    <span class="legend-item"><span class="legend-dot blocked"></span>Blocked</span>
    <span class="legend-item"><span class="legend-dot skipped"></span>Skipped</span>
  `;
  planTreeView.appendChild(legend);
  renderPlanTreeGraph(plan);

  planPrevButton.disabled = viewState.planSnapshotIndex <= 0;
  planNextButton.disabled = viewState.planSnapshotIndex >= snapshots.length - 1;
}

function renderSnapshots(hops, finalState, finalMemory) {
  viewState.snapshots = hops.map((hop) => ({
    id: `hop-${hop.hop}`,
    label: `Hop ${hop.hop}`,
    state: hop.state || {},
    memory: hop.working_memory_state || {},
  }));

  viewState.snapshots.push({
    id: "final",
    label: "Final",
    state: finalState || {},
    memory: finalMemory || {},
  });

  snapshotPicker.innerHTML = "";
  for (const snapshot of viewState.snapshots) {
    const option = document.createElement("option");
    option.value = snapshot.id;
    option.textContent = snapshot.label;
    snapshotPicker.appendChild(option);
  }

  snapshotPicker.value = "final";
  renderSelectedSnapshot();
}

function renderSelectedSnapshot() {
  const selected = viewState.snapshots.find((item) => item.id === snapshotPicker.value);
  if (!selected) {
    stateJson.textContent = "{}";
    memoryJson.textContent = "{}";
    return;
  }
  stateJson.textContent = JSON.stringify(selected.state, null, 2);
  memoryJson.textContent = JSON.stringify(selected.memory, null, 2);
}

function flattenState(input, prefix = "", out = {}) {
  if (input === null || input === undefined) {
    out[prefix || "$"] = input;
    return out;
  }
  if (typeof input !== "object") {
    out[prefix || "$"] = input;
    return out;
  }
  if (Array.isArray(input)) {
    out[prefix || "$"] = JSON.stringify(input);
    return out;
  }
  for (const [key, value] of Object.entries(input)) {
    const nextKey = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      flattenState(value, nextKey, out);
    } else {
      out[nextKey] = value;
    }
  }
  return out;
}

function computeStateDiff(prevState, nextState) {
  const prevFlat = flattenState(prevState || {});
  const nextFlat = flattenState(nextState || {});
  const added = [];
  const removed = [];
  const changed = [];

  for (const [key, value] of Object.entries(nextFlat)) {
    if (!(key in prevFlat)) {
      added.push({ key, value });
      continue;
    }
    if (JSON.stringify(prevFlat[key]) !== JSON.stringify(value)) {
      changed.push({ key, before: prevFlat[key], after: value });
    }
  }

  for (const [key, value] of Object.entries(prevFlat)) {
    if (!(key in nextFlat)) {
      removed.push({ key, value });
    }
  }

  return { added, removed, changed };
}

function deepMerge(target, patch) {
  const base = target && typeof target === "object" ? target : {};
  const incoming = patch && typeof patch === "object" ? patch : {};
  const out = Array.isArray(base) ? [...base] : { ...base };
  for (const [key, value] of Object.entries(incoming)) {
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      out[key] &&
      typeof out[key] === "object" &&
      !Array.isArray(out[key])
    ) {
      out[key] = deepMerge(out[key], value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function extractNodeEntries(hops) {
  const entries = [];
  let seq = 0;
  for (const hop of hops || []) {
    const events = Array.isArray(hop.events) ? hop.events : [];
    let priorNodeState = {};
    for (let i = 0; i < events.length; i += 1) {
      const event = events[i] || {};
      if (event.event !== "node_started") continue;

      seq += 1;
      const nodeName = String(event.node_name || "unknown");
      const beforeState = event.state && typeof event.state === "object" ? event.state : {};
      let nodeOutput = {};
      for (let j = i + 1; j < events.length; j += 1) {
        const candidate = events[j] || {};
        if (candidate.event === "node_state" && String(candidate.node_name || "") === nodeName) {
          nodeOutput = candidate;
          break;
        }
      }

      const update = nodeOutput.state_update && typeof nodeOutput.state_update === "object" ? nodeOutput.state_update : {};
      const afterState = deepMerge(beforeState, update);
      const diff = computeStateDiff(priorNodeState, afterState);
      entries.push({
        id: `node-${seq}`,
        label: `${seq}. ${nodeName}`,
        hop: hop.hop,
        nodeName,
        state_before: beforeState,
        state_after: afterState,
        state_update: update,
        output: nodeOutput,
        humanMessage: String(nodeOutput.human_message || "").trim(),
        diff,
      });
      priorNodeState = afterState;
    }
  }
  return entries;
}

function pickFirstNonEmptyString(values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value;
  }
  return "";
}

function extractNodePromptResponse(entry) {
  const raw = (entry && entry.output && typeof entry.output === "object") ? entry.output : {};
  const before = (entry && entry.state_before && typeof entry.state_before === "object") ? entry.state_before : {};
  const after = (entry && entry.state_after && typeof entry.state_after === "object") ? entry.state_after : {};
  const update = (entry && entry.state_update && typeof entry.state_update === "object") ? entry.state_update : {};

  const prompt = pickFirstNonEmptyString([
    raw.prompt,
    raw.user_prompt,
    raw.input_prompt,
    raw.state_before && raw.state_before.prompt,
    before.prompt,
    before.user_input,
    before.last_user_input,
  ]);

  const systemPrompt = pickFirstNonEmptyString([
    raw.system_prompt,
    before.system_prompt,
    after.system_prompt,
  ]);

  const response = pickFirstNonEmptyString([
    raw.response,
    raw.output_text,
    raw.human_message,
    update.response,
    after.response,
  ]);

  const llmResponse = pickFirstNonEmptyString([
    raw.raw_response,
    raw.llm_response,
    raw.generated_text,
    update.generated_text,
    after.generated_text,
  ]);

  const messages = Array.isArray(raw.messages) ? raw.messages : [];
  const toolCalls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];

  return {
    node_name: String(entry.nodeName || "unknown"),
    hop: Number(entry.hop || 0),
    prompt: prompt || null,
    system_prompt: systemPrompt || null,
    response: response || null,
    llm_response: llmResponse || null,
    messages: messages.length ? messages : null,
    tool_calls: toolCalls.length ? toolCalls : null,
  };
}

function renderSelectedNodeEntry() {
  const selected = viewState.nodeEntries.find((entry) => entry.id === viewState.selectedNodeId);
  if (!selected) {
    nodePromptJson.textContent = "{}";
    nodeOutputJson.textContent = "{}";
    nodeDiffJson.textContent = "{}";
    return;
  }
  nodePromptJson.textContent = JSON.stringify(extractNodePromptResponse(selected), null, 2);
  nodeOutputJson.textContent = JSON.stringify(
    {
      human_message: selected.humanMessage || null,
      state_update: selected.state_update || {},
      raw_event: selected.output || {},
    },
    null,
    2,
  );
  nodeDiffJson.textContent = JSON.stringify(selected.diff || {}, null, 2);
}

function renderNodeExplorer(hops) {
  const entries = extractNodeEntries(hops || []);
  viewState.nodeEntries = entries;
  nodesExplored.innerHTML = "";

  if (!entries.length) {
    nodesExplored.textContent = "No nodes explored yet.";
    nodePromptJson.textContent = "{}";
    nodeOutputJson.textContent = "{}";
    nodeDiffJson.textContent = "{}";
    return;
  }

  if (!viewState.selectedNodeId || !entries.some((entry) => entry.id === viewState.selectedNodeId)) {
    viewState.selectedNodeId = entries[entries.length - 1].id;
  }

  for (const entry of entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `node-entry ${entry.id === viewState.selectedNodeId ? "active" : ""}`;
    const suffix = entry.humanMessage ? ` - ${entry.humanMessage}` : "";
    button.textContent = `Hop ${entry.hop} | ${entry.label}${suffix}`;
    button.addEventListener("click", () => {
      viewState.selectedNodeId = entry.id;
      renderNodeExplorer(hops);
    });
    nodesExplored.appendChild(button);
  }

  renderSelectedNodeEntry();
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Running..." : "Run Turn";
}

function updateLlmStatus(llmMeta) {
  if (!llmMeta) {
    llmStatusEl.textContent = "LLM status: unknown";
    return;
  }

  if (llmMeta.startup_error) {
    llmStatusEl.textContent = `LLM status: fallback mode (startup error: ${llmMeta.startup_error})`;
    return;
  }

  const provider = llmMeta.provider || "configured";
  const model = llmMeta.model_name || "(model not reported)";
  llmStatusEl.textContent = `LLM status: active via ${provider} / ${model}`;
}

function setCallStatus(voiceMeta) {
  const active = Boolean(voiceMeta && voiceMeta.active);
  viewState.callActive = active;
  if (callPulseEl) {
    callPulseEl.classList.toggle("active", active);
  }
  if (callStatusTextEl) {
    if (!voiceMeta) {
      callStatusTextEl.textContent = "No active call runtime.";
    } else if (active) {
      callStatusTextEl.textContent = `Call runtime active | user=${voiceMeta.user_code} | session=${voiceMeta.session_id} | port=${voiceMeta.port}`;
    } else {
      callStatusTextEl.textContent = "No active call runtime.";
    }
  }
  if (callOpenLinkEl) {
    const url = voiceMeta && voiceMeta.client_url ? String(voiceMeta.client_url) : "";
    if (active && url) {
      callOpenLinkEl.href = url;
      callOpenLinkEl.classList.remove("disabled");
    } else {
      callOpenLinkEl.href = "#";
      callOpenLinkEl.classList.add("disabled");
    }
  }
}

function buildLiveFinalState(conversationState, sessionId) {
  const raw = conversationState && typeof conversationState === "object" ? conversationState : {};
  const plan = raw.active_conversation_plan && typeof raw.active_conversation_plan === "object"
    ? raw.active_conversation_plan
    : null;
  return {
    session_id: sessionId,
    response: String(raw.last_agent_response || ""),
    response_target: String(raw.last_response_target || "customer"),
    conversation_phase: "voice_call_live",
    conversation_plan: plan || undefined,
    active_conversation_plan: plan || undefined,
    ...raw,
  };
}

function buildLiveHopsFromTrace(tracePayload, finalState, workingMemoryState) {
  if (!tracePayload || typeof tracePayload !== "object") return [];
  const summary = tracePayload.summary && typeof tracePayload.summary === "object" ? tracePayload.summary : {};
  const nodeHistory = Array.isArray(summary.node_hits) ? summary.node_hits : [];
  const nodes = Array.isArray(tracePayload.nodes) ? tracePayload.nodes : [];
  const toolCalls = Array.isArray(tracePayload.tool_calls) ? tracePayload.tool_calls : [];
  const events = [];
  for (const node of nodes) {
    events.push({
      event: "node_finished",
      node_name: String(node.node_name || "unknown"),
      status: String(node.status || "completed"),
      duration_ms: typeof node.duration_ms === "number" ? node.duration_ms : null,
    });
  }
  for (const tool of toolCalls) {
    events.push({
      event: "tool_call",
      tool_name: String(tool.tool_name || "unknown_tool"),
      status: String(tool.status || "completed"),
      duration_ms: typeof tool.duration_ms === "number" ? tool.duration_ms : null,
    });
  }
  const thinking = [
    `trace: ${String(tracePayload.trace_id || "-")}`,
    `status: ${String(tracePayload.status || "unknown")}`,
    `nodes: ${nodeHistory.join(" -> ") || "(none)"}`,
  ];
  return [
    {
      hop: 1,
      input: String(tracePayload.user_input || ""),
      response: String(finalState.response || ""),
      response_target: String(finalState.response_target || "customer"),
      elapsed_ms: null,
      node_history: nodeHistory,
      conversation_phase: "voice_call_live",
      thinking,
      events,
      trace_summary: summary,
      state: finalState,
      working_memory_state: workingMemoryState || {},
    },
  ];
}

function renderCallTranscript(messages) {
  if (!callTranscriptEl) return;
  callTranscriptEl.innerHTML = "";
  if (!Array.isArray(messages) || !messages.length) {
    callTranscriptEl.textContent = "No transcript yet.";
    return;
  }
  for (const message of messages) {
    const role = String(message.role || "unknown");
    const content = String(message.content || "").trim();
    if (!content) continue;
    const line = document.createElement("div");
    line.className = `call-line ${role}`;
    line.textContent = `${role}: ${content}`;
    callTranscriptEl.appendChild(line);
  }
  callTranscriptEl.scrollTop = callTranscriptEl.scrollHeight;
}

async function refreshCallTranscript(sessionId) {
  const sid = String(sessionId || "").trim();
  if (!sid) return;
  try {
    const response = await fetch(`/api/session/${encodeURIComponent(sid)}/messages?limit=120`);
    if (!response.ok) return;
    const payload = await response.json();
    renderCallTranscript(payload.messages || []);
  } catch (_error) {
    // Keep polling resilient without interrupting main UI flow.
  }
}

function startTranscriptPolling(sessionId) {
  const sid = String(sessionId || "").trim();
  if (!sid) return;
  viewState.callSessionId = sid;
  if (viewState.transcriptPoller) {
    window.clearInterval(viewState.transcriptPoller);
    viewState.transcriptPoller = null;
  }
  refreshCallTranscript(sid);
  viewState.transcriptPoller = window.setInterval(() => {
    refreshCallTranscript(sid);
  }, 1500);
}

function stopTranscriptPolling() {
  if (viewState.transcriptPoller) {
    window.clearInterval(viewState.transcriptPoller);
    viewState.transcriptPoller = null;
  }
}

function stopLiveStatePolling() {
  if (viewState.liveStatePoller) {
    window.clearInterval(viewState.liveStatePoller);
    viewState.liveStatePoller = null;
  }
}

async function refreshCallLiveState(sessionId) {
  const sid = String(sessionId || "").trim();
  if (!sid) return;
  try {
    const [sessionResponse, traceResponse] = await Promise.all([
      fetch(`/api/session/${encodeURIComponent(sid)}`),
      fetch(`/api/session/${encodeURIComponent(sid)}/latest-trace`),
    ]);
    if (!sessionResponse.ok) return;

    const sessionPayload = await sessionResponse.json();
    const tracePayload = traceResponse.ok ? await traceResponse.json() : { trace: null };
    const finalState = buildLiveFinalState(sessionPayload.conversation_state || {}, sid);
    const finalMemory = sessionPayload.working_memory_state || {};
    const liveHops = buildLiveHopsFromTrace(tracePayload.trace, finalState, finalMemory);

    if (liveHops.length) {
      renderExecutionChart(liveHops);
      renderExecutionNarrative(liveHops);
      renderThinking(liveHops);
      renderNodeExplorer(liveHops);
      viewState.planSnapshots = extractPlanSnapshots(liveHops, finalState);
    } else {
      renderExecutionChart([]);
      renderExecutionNarrative([]);
      renderThinking([]);
      renderNodeExplorer([]);
      viewState.planSnapshots = extractPlanSnapshots([], finalState);
    }
    viewState.planSnapshotIndex = Math.max(0, viewState.planSnapshots.length - 1);
    renderPlanTree();
    renderSnapshots(liveHops, finalState, finalMemory);
  } catch (_error) {
    // Keep live polling resilient.
  }
}

function startLiveStatePolling(sessionId) {
  const sid = String(sessionId || "").trim();
  if (!sid) return;
  stopLiveStatePolling();
  refreshCallLiveState(sid);
  viewState.liveStatePoller = window.setInterval(() => {
    refreshCallLiveState(sid);
  }, 1500);
}

async function refreshVoiceStatus() {
  try {
    const response = await fetch("/api/voice/status");
    if (!response.ok) return;
    const payload = await response.json();
    setCallStatus(payload);
    if (payload && payload.active && payload.session_id) {
      startTranscriptPolling(String(payload.session_id));
      startLiveStatePolling(String(payload.session_id));
    } else if (!payload || !payload.active) {
      stopTranscriptPolling();
      stopLiveStatePolling();
    }
  } catch (_error) {
    // Ignore transient poll errors.
  }
}

function renderTurnPayload(payload) {
  renderExecutionChart(payload.hops || []);
  renderExecutionNarrative(payload.hops || []);
  renderThinking(payload.hops || []);
  viewState.planSnapshots = extractPlanSnapshots(payload.hops || [], payload.final_state || {});
  viewState.planSnapshotIndex = Math.max(0, viewState.planSnapshots.length - 1);
  renderPlanTree();
  renderNodeExplorer(payload.hops || []);
  renderSnapshots(
    payload.hops || [],
    payload.final_state || {},
    payload.final_working_memory_state || {},
  );
  updateLlmStatus(payload.llm || null);
}

function createLiveHop(hop, input) {
  return {
    hop,
    input,
    response: "",
    response_target: "pending",
    elapsed_ms: null,
    node_history: [],
    conversation_phase: "running",
    thinking: [],
    events: [],
    trace_summary: {},
    state: {},
    working_memory_state: {},
  };
}

function findOrCreateLiveHop(hop, input = "") {
  let existing = viewState.liveHops.find((item) => Number(item.hop) === Number(hop));
  if (!existing) {
    existing = createLiveHop(hop, input);
    viewState.liveHops.push(existing);
    viewState.liveHops.sort((a, b) => Number(a.hop) - Number(b.hop));
  }
  return existing;
}

function refreshLivePanels() {
  renderExecutionChart(viewState.liveHops);
  renderExecutionNarrative(viewState.liveHops);
  renderThinking(viewState.liveHops);
  viewState.planSnapshots = extractPlanSnapshots(viewState.liveHops, {});
  viewState.planSnapshotIndex = Math.max(0, viewState.planSnapshots.length - 1);
  renderPlanTree();
  renderNodeExplorer(viewState.liveHops);
  renderSnapshots(viewState.liveHops, {}, {});
}

async function runUserTurn(message) {
  const sessionId = sessionIdInput.value.trim() || "collection-ui-session";
  addBubble("user", message);
  setBusy(true);
  viewState.liveHops = [];
  refreshLivePanels();

  await new Promise((resolve) => {
    const params = new URLSearchParams({
      message,
      session_id: sessionId,
      soft_cap: "10",
      hard_cap: "50",
    });
    const source = new EventSource(`/api/run-turn-stream?${params.toString()}`);
    let finished = false;

    source.addEventListener("hop_started", (evt) => {
      const payload = JSON.parse(evt.data || "{}");
      const hop = Number(payload.hop || 0);
      if (hop > 0) {
        findOrCreateLiveHop(hop, String(payload.input || ""));
        refreshLivePanels();
      }
    });

    source.addEventListener("trace_event", (evt) => {
      const payload = JSON.parse(evt.data || "{}");
      const hop = Number(payload.hop || 0);
      const traceEvent = payload.trace_event || {};
      if (hop > 0) {
        const liveHop = findOrCreateLiveHop(hop);
        liveHop.events.push(traceEvent);
        refreshLivePanels();
      }
    });

    source.addEventListener("hop_update", (evt) => {
      const payload = JSON.parse(evt.data || "{}");
      const hop = Number(payload.hop || 0);
      if (hop > 0) {
        const idx = viewState.liveHops.findIndex((item) => Number(item.hop) === hop);
        if (idx >= 0) {
          viewState.liveHops[idx] = payload;
        } else {
          viewState.liveHops.push(payload);
          viewState.liveHops.sort((a, b) => Number(a.hop) - Number(b.hop));
        }
        refreshLivePanels();
      }
    });

    source.addEventListener("turn_complete", (evt) => {
      const payload = JSON.parse(evt.data || "{}");
      finished = true;
      source.close();
      addBubble("agent", payload.final_response || "No response generated.");
      renderTurnPayload(payload);
      setBusy(false);
      resolve();
    });

    source.addEventListener("turn_error", (evt) => {
      const payload = JSON.parse(evt.data || "{}");
      finished = true;
      source.close();
      addBubble("agent", `Error: ${String(payload.error || "Unknown error")}`);
      setBusy(false);
      resolve();
    });

    source.addEventListener("stream_close", () => {
      if (!finished) {
        source.close();
        setBusy(false);
        resolve();
      }
    });

    source.onerror = () => {
      if (!finished) {
        source.close();
        addBubble("agent", "Error: live stream disconnected.");
        setBusy(false);
        resolve();
      }
    };
  });
}

function renderDemoUsers(users) {
  demoUsersEl.innerHTML = "";

  for (const entry of users) {
    const customer = entry.customer || {};
    const caseInfo = entry.case || {};
    const card = document.createElement("div");
    card.className = "demo-user-card";

    card.innerHTML = `
      <h3>${entry.display_name}</h3>
      <p class="demo-user-meta"><strong>Name:</strong> ${customer.name}</p>
      <p class="demo-user-meta"><strong>Customer ID:</strong> ${customer.customer_id}</p>
      <p class="demo-user-meta"><strong>Case:</strong> ${caseInfo.case_id} (DPD ${caseInfo.dpd})</p>
      <p class="demo-user-meta"><strong>DOB:</strong> ${customer.dob} | <strong>ZIP:</strong> ${customer.zip}</p>
      <p class="demo-user-meta"><strong>PAN last4:</strong> ${customer.last4_pan}</p>
      <p class="demo-user-meta"><strong>Overdue:</strong> ${caseInfo.overdue_amount} | <strong>EMI:</strong> ${caseInfo.emi_amount}</p>
      <div class="demo-user-actions">
        <button type="button" data-action="chat" data-user-code="${entry.user_code}">Start Chat</button>
        <button type="button" data-action="reset" data-user-code="${entry.user_code}">Reset Chat</button>
        <button type="button" data-action="call" data-user-code="${entry.user_code}">Start Call</button>
      </div>
    `;

    const chatButton = card.querySelector("button[data-action='chat']");
    const resetButton = card.querySelector("button[data-action='reset']");
    const callButton = card.querySelector("button[data-action='call']");
    chatButton.addEventListener("click", () => startDemoConversation(entry));
    resetButton.addEventListener("click", () => resetDemoConversation(entry));
    callButton.addEventListener("click", () => startDemoCall(entry));
    demoUsersEl.appendChild(card);
  }
}

async function loadDemoUsers() {
  try {
    const response = await fetch("/api/demo-users");
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    viewState.demoUsers = Array.isArray(payload.users) ? payload.users : [];
    renderDemoUsers(viewState.demoUsers);
  } catch (error) {
    demoUsersEl.textContent = `Failed to load demo users: ${String(error)}`;
  }
}

async function startDemoConversation(userEntry) {
  const sessionId = `collection-${userEntry.user_code}`;
  sessionIdInput.value = sessionId;
  setBusy(true);

  try {
    const response = await fetch("/api/start-conversation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_code: userEntry.user_code,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    if (payload.error) {
      throw new Error(payload.error);
    }

    addBubble(
      "system",
      `Initialized ${userEntry.display_name} | case=${userEntry.case.case_id} | customer_id=${userEntry.customer.customer_id}`,
    );

    const turn = payload.turn || {};
    addBubble("agent", turn.final_response || "No opener generated.");
    renderTurnPayload(turn);
  } catch (error) {
    addBubble("agent", `Error starting demo conversation: ${String(error)}`);
  } finally {
    setBusy(false);
  }
}

async function resetDemoConversation(userEntry) {
  const sessionId = `collection-${userEntry.user_code}`;
  sessionIdInput.value = sessionId;
  setBusy(true);

  try {
    const response = await fetch("/api/reset-conversation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_code: userEntry.user_code,
        session_id: sessionId,
      }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    if (payload.error) {
      throw new Error(payload.error);
    }

    if (String(viewState.sessionId || "") === sessionId) {
      viewState.events = [];
      viewState.snapshots = [];
      viewState.snapshotMap = new Map();
      viewState.planSnapshots = [];
      viewState.planSnapshotIndex = 0;
      messagesEl.innerHTML = "";
      renderSnapshots([], {}, {});
      renderPlanTree();
      renderExecutionChart([]);
      renderExecutionNarrative([]);
      renderThinking([]);
      renderNodeExplorer([]);
    }

    addBubble(
      "system",
      `Reset conversation for ${userEntry.display_name} | session=${sessionId}. Next Start Chat will begin from fresh verification flow.`,
    );
  } catch (error) {
    addBubble("agent", `Error resetting demo conversation: ${String(error)}`);
  } finally {
    setBusy(false);
  }
}

async function startDemoCall(userEntry) {
  const sessionId = `collection-${userEntry.user_code}`;
  sessionIdInput.value = sessionId;
  setBusy(true);

  try {
    const response = await fetch("/api/start-call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_code: userEntry.user_code,
        session_id: sessionId,
        transport: "webrtc",
        port: 8788,
      }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (payload.error) throw new Error(payload.error);

    addBubble(
      "system",
      `Started call mode for ${userEntry.display_name} | session=${sessionId}`,
    );
    if (payload.turn && payload.turn.final_response) {
      addBubble("agent", String(payload.turn.final_response));
      renderTurnPayload(payload.turn);
    }

    const voice = payload.voice || {};
    setCallStatus(voice);
    if (voice.active && voice.session_id) {
      startTranscriptPolling(String(voice.session_id));
    }
    if (voice.client_url) {
      addBubble("system", `Open call client: ${String(voice.client_url)}`);
    }
  } catch (error) {
    addBubble("agent", `Error starting call mode: ${String(error)}`);
  } finally {
    setBusy(false);
  }
}

async function stopVoiceCall() {
  try {
    const response = await fetch("/api/voice/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: false }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    setCallStatus(payload);
    stopTranscriptPolling();
    stopLiveStatePolling();
    addBubble("system", "Stopped call runtime.");
  } catch (error) {
    addBubble("agent", `Error stopping call runtime: ${String(error)}`);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;
  messageInput.value = "";
  await runUserTurn(message);
});

snapshotPicker.addEventListener("change", renderSelectedSnapshot);

collapseButton.addEventListener("click", () => {
  viewState.collapsed = !viewState.collapsed;
  appShell.classList.toggle("collapsed", viewState.collapsed);
  collapseButton.textContent = viewState.collapsed ? "Expand" : "Collapse";
});

if (planPrevButton) {
  planPrevButton.addEventListener("click", () => {
    viewState.planSnapshotIndex = Math.max(0, viewState.planSnapshotIndex - 1);
    renderPlanTree();
  });
}

if (planNextButton) {
  planNextButton.addEventListener("click", () => {
    viewState.planSnapshotIndex = Math.min(
      Math.max(0, viewState.planSnapshots.length - 1),
      viewState.planSnapshotIndex + 1,
    );
    renderPlanTree();
  });
}

if (callStopButton) {
  callStopButton.addEventListener("click", async () => {
    await stopVoiceCall();
  });
}

renderExecutionChart([]);
renderExecutionNarrative([]);
renderThinking([]);
renderPlanTree();
renderNodeExplorer([]);
renderSnapshots([], {}, {});
loadDemoUsers();
refreshVoiceStatus();
window.setInterval(refreshVoiceStatus, 5000);
