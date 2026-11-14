// Project browser hand-landmark bounds into an object-fit:contain media viewport.
// These are hand-localizer bounds, not bounding boxes predicted by YOLO.
export function projectHandBox(box, sourceWidth, sourceHeight, width, height, mirrored = false) {
  if (!box || ![sourceWidth, sourceHeight, width, height].every(v => Number.isFinite(v) && v > 0) ||
      ![box.x, box.y, box.width, box.height].every(Number.isFinite) || box.width <= 0 || box.height <= 0) return null;
  const left = Math.max(0, box.x), top = Math.max(0, box.y);
  const right = Math.min(sourceWidth, box.x + box.width), bottom = Math.min(sourceHeight, box.y + box.height);
  if (right <= left || bottom <= top) return null;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const offsetX = (width - sourceWidth * scale) / 2;
  const offsetY = (height - sourceHeight * scale) / 2;
  return {left: offsetX + (mirrored ? sourceWidth - right : left) * scale,
    top: offsetY + top * scale, width: (right - left) * scale, height: (bottom - top) * scale};
}
