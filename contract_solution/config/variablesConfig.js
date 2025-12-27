import documentData from "../data/input.json";

// extract visible text from a mention-like node
function extractTextFromNode(node) {
  const childText =
    node.children?.map((child) => child.text || "").join("") || "";
  return childText || node.value || node.title || "";
}

// Recursive walk over the document tree to collect first occurrences of mentions
function collectMentions(nodes, acc) {
  if (!Array.isArray(nodes)) return;

  nodes.forEach((node) => {
    if (!node || typeof node !== "object") return;

    // If this node is a mention, record its value
    if (node.type === "mention") {
      const key = node.id || node.title;
      if (key && acc[key] == null) {
        acc[key] = extractTextFromNode(node);
      }
    }

    // Recurse into children
    if (Array.isArray(node.children)) {
      collectMentions(node.children, acc);
    }
  });
}

// Build the initial variable map once from the input.json
const VARIABLE_VALUES = (() => {
  const acc = {};
  collectMentions(documentData, acc);
  return acc;
})();

export function getMentionValue(node) {
  const key = node.id || node.title;

  if (key && VARIABLE_VALUES[key] != null) {
    return VARIABLE_VALUES[key];
  }

  return extractTextFromNode(node);
}
