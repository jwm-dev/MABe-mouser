/// <reference types="vite/client" />

// Declare CSS modules
declare module '*.css' {
  const content: Record<string, string>
  export default content
}

// Declare CSS side-effect imports
declare module '*.css' {
  const content: any
  export = content
}
