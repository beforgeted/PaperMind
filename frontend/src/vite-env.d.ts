/// <reference types="vite/client" />

/**
 * 构建时环境变量
 */
interface ImportMetaEnv {
  /** 后端基地址，如 `http://127.0.0.1:8000`；留空则走同源 + Vite 代理 */
  readonly VITE_API_BASE: string
}
