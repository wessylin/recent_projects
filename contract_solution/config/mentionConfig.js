// Explicit styles for known variables

const MENTION_CONFIG = {
  // From JSON
  "Contract Date": {
    background: "rgb(20, 170, 245)",
  },
  Provider: {
    // id: "Provider"
    background: "rgb(175, 56, 235)",
  },
  Client: {
    // id: "Client"
    background: "rgb(175, 184, 59)",
  },
  Term: {
    background: "rgb(126, 204, 73)",
  },
  "Governing Law Jurisdiction": {
    background: "rgb(250, 208, 0)",
  },
};

// A palette to build unique backgrounds for new mention types
const FALLBACK_COLORS = [
  "rgb(20, 170, 245)",   // blue
  "rgb(175, 56, 235)",   // purple
  "rgb(175, 184, 59)",   // olive
  "rgb(126, 204, 73)",   // green
  "rgb(250, 208, 0)",    // yellow
];

// Deterministic hash index into FALLBACK_COLORS
function stringToIndex(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i += 1) {
    hash = (hash * 31 + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % FALLBACK_COLORS.length;
}

export function getMentionStyle(node) {
  const key = node.id || node.title || node.variableType || "default";

  if (MENTION_CONFIG[key]) {
    return {
      background: MENTION_CONFIG[key].background,
      color: MENTION_CONFIG[key].color || "#ffffff",
    };
  }

  const background = FALLBACK_COLORS[stringToIndex(key)];

  return {
    background,
    color: "#ffffff",
  };
}
