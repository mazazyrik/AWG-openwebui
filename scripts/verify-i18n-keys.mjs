import fs from 'node:fs';

const changedFilesPath = process.argv[2];
const localePath = 'src/lib/i18n/locales/en-US/translation.json';
const translations = JSON.parse(fs.readFileSync(localePath, 'utf8'));
const changedFiles = fs.existsSync(changedFilesPath)
	? fs.readFileSync(changedFilesPath, 'utf8').split('\n').filter(Boolean)
	: [];
const missing = new Set();
const pattern = /\$i18n\.t\(\s*['"]([^'"]+)['"]/g;

for (const path of changedFiles.filter((path) => /\.(?:js|ts|svelte)$/.test(path) && fs.existsSync(path))) {
	const source = fs.readFileSync(path, 'utf8');
	for (const match of source.matchAll(pattern)) {
		if (!(match[1] in translations)) missing.add(match[1]);
	}
}

if (missing.size) {
	console.error(`Missing en-US i18n keys:\n${[...missing].sort().join('\n')}`);
	process.exit(1);
}
