import React from "react";
import { getMentionStyle } from "../config/mentionConfig";
import { getMentionValue } from "../config/variablesConfig";


export default function Mention({ node }) {
  const style = getMentionStyle(node);
  const text = getMentionValue(node); 

  return (
    <span
      className="mention-pill"
      style={{
        backgroundColor: style.background,
        color: style.color,
      }}
      title={node.title || node.id}
    >
      {text}
    </span>
  );
}
