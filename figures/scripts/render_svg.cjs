// Export an SVG figure to PDF (vector, fonts embedded) and PNG (600 dpi) with headless Chromium.
// Usage: node render_svg.cjs <figure.svg> [stix_font_dir]   (needs: npm i -g playwright; NODE_PATH=$(npm root -g))
// The optional font directory supplies STIXGeneral*.ttf when the system has no STIX fonts (macOS ships STIX Two).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
const svgPath = path.resolve(process.argv[2]);
const fontDir = process.argv[3];
const svg = fs.readFileSync(svgPath, 'utf8').replace(/<\?xml[^>]*\?>/, '');
const vb = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
const W = parseFloat(vb[1]), H = parseFloat(vb[2]);          // 10 units = 1 mm
const mmW = W / 10, mmH = H / 10;
let faces = '';
if (fontDir) {
  const f = (file, style, weight) => `@font-face{font-family:'STIXGeneral';font-style:${style};font-weight:${weight};` +
    `src:url('file://${path.join(fontDir, file)}');}`;
  faces = f('STIXGeneral.ttf', 'normal', 400) + f('STIXGeneralItalic.ttf', 'italic', 400) +
          f('STIXGeneralBol.ttf', 'normal', 700) + f('STIXGeneralBolIta.ttf', 'italic', 700);
}
const html = `<!doctype html><html><head><meta charset="utf-8"><style>${faces}
@page{size:${mmW}mm ${mmH}mm;margin:0}html,body{margin:0;padding:0;background:#fff}
svg{display:block;width:${mmW}mm;height:${mmH}mm}</style></head><body>${svg}</body></html>`;
const tmp = svgPath.replace(/\.svg$/, '.render.html');
fs.writeFileSync(tmp, html);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: Math.ceil(mmW / 25.4 * 96), height: Math.ceil(mmH / 25.4 * 96) },
                                     deviceScaleFactor: 600 / 96 });
await page.goto('file://' + tmp);
await page.evaluate(() => document.fonts.ready);
await page.pdf({ path: svgPath.replace(/\.svg$/, '.pdf'), width: `${mmW}mm`, height: `${mmH}mm`, printBackground: true,
                 pageRanges: '1' });
await page.locator('svg').screenshot({ path: svgPath.replace(/\.svg$/, '.png') });
await browser.close();
fs.unlinkSync(tmp);
console.log('done', mmW, 'x', mmH, 'mm');
})();
