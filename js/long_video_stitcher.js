import { app } from "../../scripts/app.js";

// Dynamic latent inputs for the Long Video Stitcher node.
//
// The backend declares latent_1..latent_12 (all optional except latent_1). This
// extension keeps only the connected latent slots plus ONE spare empty slot
// visible, so the node "grows" as you connect chunks and "shrinks" when you
// disconnect them — up to the backend cap (12).
//
// Everything is wrapped in try/catch: if anything goes wrong the node simply
// falls back to showing all 12 static slots, which still works.

const MAX_LATENTS = 12;
const MIN_VISIBLE = 2; // always show at least latent_1 + one spare
// Nodes that expose dynamic latent_1..latent_12 inputs.
const DYNAMIC_LATENT_NODES = ["LongVideoStitcher", "SmoothVideoStitcher", "SmoothAudioStitcher"];

function latentIndex(name) {
    const m = /^latent_(\d+)$/.exec(name || "");
    return m ? parseInt(m[1], 10) : null;
}

function syncLatentInputs(node) {
    try {
        const inputs = node.inputs || [];

        // Highest latent index that currently has a connection.
        let maxConnected = 0;
        for (const inp of inputs) {
            const idx = latentIndex(inp.name);
            if (idx != null && inp.link != null && idx > maxConnected) maxConnected = idx;
        }

        // Show connected slots + one spare, clamped to [MIN_VISIBLE, MAX_LATENTS].
        const desired = Math.min(MAX_LATENTS, Math.max(MIN_VISIBLE, maxConnected + 1));

        // How many latent slots are visible now?
        let visible = 0;
        for (const inp of inputs) if (latentIndex(inp.name) != null) visible++;

        // Add missing trailing slots.
        while (visible < desired) {
            visible++;
            node.addInput(`latent_${visible}`, "LATENT");
        }

        // Remove trailing UNCONNECTED slots above `desired` (never touch a linked slot
        // or latent_1).
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            const idx = latentIndex(inp.name);
            if (idx == null) continue;
            if (idx > desired && idx > 1 && inp.link == null) {
                node.removeInput(i);
            }
        }

        if (node.graph) app.graph.setDirtyCanvas(true, true);
    } catch (e) {
        console.error("[LongVideoStitcher] dynamic input sync failed (falling back to static slots):", e);
    }
}

app.registerExtension({
    name: "Comfy.LongVideoStitcher.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!DYNAMIC_LATENT_NODES.includes(nodeData.name)) return;

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slotIndex, isConnected, link, ioSlot) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            // Re-sync after the connection has settled.
            try {
                if (slotType === 1 /* INPUT */) setTimeout(() => syncLatentInputs(this), 0);
            } catch (e) { /* ignore */ }
            return r;
        };
    },
    async nodeCreated(node) {
        if (!DYNAMIC_LATENT_NODES.includes(node.comfyClass)) return;
        // Collapse to the minimal slot set on fresh nodes, and re-sync after a loaded
        // graph has restored its links.
        setTimeout(() => syncLatentInputs(node), 0);
    },
});
