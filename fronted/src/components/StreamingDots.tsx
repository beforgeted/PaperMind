/**
 * 流式生成中的三点跳动指示器。
 */
export function StreamingDots() {
  return (
    <span className="streaming-dots" aria-label="正在生成">
      <span />
      <span />
      <span />
    </span>
  )
}
