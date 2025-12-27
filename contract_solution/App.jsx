import React from "react";
import documentData from "./data/input.json";
import { renderNodes } from "./renderer";

function App() {
  const rootBlock = documentData[0];
  const contentNodes = rootBlock.children || [];

  return (
    <div className="app-shell">
      <div className="doc-card">
        {renderNodes(contentNodes)}
      </div>
    </div>
  );
}


export default App;

