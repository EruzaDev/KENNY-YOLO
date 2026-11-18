// Apply the actual browser guard to model-generated validation landmarks.
import {readFile, writeFile} from 'node:fs/promises';
const source = await readFile('frontend/js/hand_gate.js');
const {okPinchEvidence} = await import(`data:text/javascript;base64,${source.toString('base64')}`);
const input = JSON.parse(await readFile('outputs/ok_bias/validation_landmarks.json', 'utf8'));
const predictions = input.predictions.map(row => ({...row, evidence:okPinchEvidence(row.world_landmarks)}));
const summary = {};
for (const row of predictions) {
  summary[row.label] ||= {};
  summary[row.label][row.evidence] = (summary[row.label][row.evidence] || 0) + 1;
}
const report = {kind:'Browser guard on predicted validation landmarks; not camera performance', split:'val',
  valid_ok_vetoes:predictions.filter(row => row.label==='ok' && row.evidence==='separate').length, summary, predictions};
await writeFile('outputs/ok_bias/ok_guard_validation.json', JSON.stringify(report,null,2));
console.log(JSON.stringify({valid_ok_vetoes:report.valid_ok_vetoes,summary},null,2));
if (report.valid_ok_vetoes) process.exitCode = 1;
