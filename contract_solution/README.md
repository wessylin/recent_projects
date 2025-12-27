A contract UI in React that dynamically builds Service Agreements by parsing data from an input.json file. <br>

- **Tools**: React, Vite
- **Deployment**: 'npm run dev' <br>

--------------------
**Key Design Features**
1. **Comprehensive Markdown**: Implemented context-based mark propagation so styling now cascades cleanly through nested blocks. (renderer.jsx) 
2. **Scalable Mention Styling**: Added a deterministic fallback color system for mentions, so new mention types are automatically supported. (Mention.jsx, mentionConfig.js)
3. **Linking Pills via a Variable Map**: Introduced an auto-generated variable map that binds all mentions referencing the same variable, so content is in sync and simplifies future editing. (variablesConfig.js)

<img width="1125" height="745" alt="ouput" src="https://github.com/user-attachments/assets/a3829914-4834-43ef-87e2-acce04ce050b" />

