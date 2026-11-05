const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

async function captureScreens() {
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1400,900']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  // 1. Capture de la liste des runs et métriques
  console.log('Navigating to MLflow experiments page...');
  await page.goto('http://localhost:5000/#/experiments/1', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2500));
  await page.screenshot({ path: path.resolve('reports/mlflow_runs_list.png') });

  // 2. Capture du run détaillé avec paramètres et version DVC
  const runId = 'aab490b3a0624e48bcd287117837eb56';
  console.log(`Navigating to run details: ${runId}...`);
  await page.goto(`http://localhost:5000/#/experiments/1/runs/${runId}`, { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2500));
  await page.screenshot({ path: path.resolve('reports/mlflow_dvc_tracking.png') });

  // 3. Capture de la section Artefacts (défiler vers le bas)
  console.log('Scrolling to artifacts section...');
  await page.evaluate(() => window.scrollBy(0, 650));
  await new Promise(r => setTimeout(r, 1500));
  await page.screenshot({ path: path.resolve('reports/mlflow_artifacts.png') });

  await browser.close();
  console.log('All 3 screenshots captured successfully!');
}

captureScreens().catch(console.error);
