import React from "react";
import Mention from "./components/Mention.jsx";

function renderInline(node, key, ctx = {}) {
  if (node.type === "mention") {
    return <Mention key={key} node={node} />;
  }

  const style = {};
  if (node.color && !node.type) {
    style.color = node.color;
  }

  // Inherit marks from context if not set on the node itself
  const appliedBold = node.bold || ctx.bold;
  const appliedUnderline = node.underline || ctx.underline;
  const appliedItalic = node.italic || ctx.italic;

  let className = "";
  if (appliedBold) className += " bold";
  if (appliedUnderline) className += " underline";
  if (appliedItalic) className += " italic";

  return (
    <span key={key} className={className} style={style}>
      {node.text || ""}
    </span>
  );
}

export function renderNode(node, key, ctx = {}) {
  if (!node) return null;
  const children = node.children || [];
  
  // Merge this node's marks into the context, so descendants inherit them

  const markCtx = {
  ...ctx,
  bold: node.bold || ctx.bold,
  underline: node.underline || ctx.underline,
  italic: node.italic || ctx.italic,
  };

  switch (node.type) {
    /* ---- blocks & clauses ---- */
    case "block":
      return (
        <div key={key} className="section-block">
          {children.map((child, i) =>
            renderNode(child, `${key}-${i}`, markCtx)
          )}
        </div>
      );

    case "clause": {
      // If we have a subClauseLabel (a, b, c...), this is a lettered child clause
      if (ctx.subClauseLabel) {
        return renderLetteredSubClause(node, key, markCtx.subClauseLabel, markCtx);
      }

      // Otherwise this is a numbered top-level clause (1., 2., 3.)
      return renderNumberedClause(node, key, markCtx.clauseNumber, markCtx);
    }

    /* ---- headings ---- */
    case "h1":
      return (
        <h1 key={key}>
          {children.map((child, i) =>
            renderInline(child, `${key}-h1-${i}`, markCtx)
          )}
        </h1>
      );

    case "h4":
      return (
        <h4 key={key}>
          {children.map((child, i) =>
            renderInline(child, `${key}-h4-${i}`, markCtx)
          )}
        </h4>
      );

    /* ---- paragraphs ---- */
    case "p": {
      const inlineParts = [];
      const blockParts = [];

      children.forEach((child, i) => {
        if (child.type === "clause") {
          // If a clause lives inside a paragraph use the passed-in clause number for that child.
          const childCtx = {...markCtx};
          if (markCtx.clauseNumberForChild) {
            childCtx.clauseNumber = markCtx.clauseNumberForChild;
          }
          blockParts.push(
            renderNode(child, `${key}-pclause-${i}`, childCtx)
          );
        } else if (child.type === "block") {
          blockParts.push(
            renderNode(child, `${key}-pblock-${i}`, markCtx)
          );
        } else {
          inlineParts.push(
            renderInline(child, `${key}-pinline-${i}`, markCtx)
          );
        }
      });

      return (
        <div key={key}>
          {inlineParts.length > 0 && <p>{inlineParts}</p>}
          {blockParts}
        </div>
      );
    }

    case "ul":
      return (
        <ul key={key}>
          {children.map((child, i) =>
            renderNode(child, `${key}-li-${i}`, markCtx)
          )}
        </ul>
      );

    case "li":
      return (
        <li key={key}>
          {children.map((child, i) =>
            renderNode(child, `${key}-lic-${i}`, markCtx)
          )}
        </li>
      );

    case "lic":
      return (
        <span key={key}>
          {children.map((child, i) =>
            renderInline(child, `${key}-lic-${i}`, markCtx)
          )}
        </span>
      );

    default:
      if (node.text) return renderInline(node, key, markCtx);
      return null;
  }
}

/* ---------- helpers for clauses ---------- */

// Top-level numbered clause
function renderNumberedClause(node, key, clauseNumber, ctx={}) {
  const children = node.children || [];
  let letterIndex = 0; 

  const markCtx = {
  ...ctx,
  bold: node.bold || ctx.bold,
  underline: node.underline || ctx.underline,
  italic: node.italic || ctx.italic,
  };

  return (
    <div key={key} className="section-block clause-block">
      {children.map((child, i) => {
        // Heading inside clause
        if (child.type === "h4") {
          return (
            <h4 key={`${key}-h4-${i}`}>
              {clauseNumber && (
                <span className="clause-number">{clauseNumber}. </span>
              )}
              {child.children?.map((c, j) =>
                renderInline(c, `${key}-h4-inline-${j}`, markCtx)
              )}
            </h4>
          );
        }

        // Child clauses → lettered subclauses
        if (child.type === "clause") {
          const label = String.fromCharCode(
            "a".charCodeAt(0) + letterIndex
          );
          letterIndex += 1;
          return renderNode(child, `${key}-sub-${i}`, {
            ...markCtx,
            subClauseLabel: label,
          });
        }

        return renderNode(child, `${key}-${i}`, markCtx);
      })}
    </div>
  );
}

// Lettered subclause
function renderLetteredSubClause(node, key, label, ctx={}) {
  const children = node.children || [];

  const markCtx = {
  ...ctx,
  bold: node.bold || ctx.bold,
  underline: node.underline || ctx.underline,
  italic: node.italic || ctx.italic,
  };

  return (
    <div key={key} className="subclause">
      {children.map((child, i) => {
        if (child.type === "p") {
          const pChildren = child.children || [];
          return (
            <p key={`${key}-p-${i}`}>
              <span className="subclause-label">({label}) </span>
              {pChildren.map((c, j) =>
                renderInline(c, `${key}-sub-inline-${j}`, markCtx)
              )}
            </p>
          );
        }
        return renderNode(child, `${key}-${i}`, markCtx);
      })}
    </div>
  );
}

/* ---------- helper that assigns numbers ---------- */

export function renderNodes(nodes) {
  let clauseCounter = 0;
  const elements = [];

  nodes.forEach((node, idx) => {
    // Direct top-level clause: number it
    if (node.type === "clause") {
      clauseCounter += 1;
      elements.push(
        renderNode(node, `n-${idx}`, { clauseNumber: clauseCounter })
      );
      return;
    }

    // Paragraph that contains a clause as its only child
    if (node.type === "p") {
      const children = node.children || [];
      if (children.length === 1 && children[0].type === "clause") {
        clauseCounter += 1;
        elements.push(
          renderNode(node, `n-${idx}`, {
            clauseNumberForChild: clauseCounter,
          })
        );
        return;
      }
    }

    // Anything else: render normally
    elements.push(renderNode(node, `n-${idx}`, {}));
  });

  return elements;
}
