import fs from 'node:fs';

const changedFilesPath = process.argv[2];
const localePath = 'src/lib/i18n/locales/en-US/translation.json';
const translations = JSON.parse(fs.readFileSync(localePath, 'utf8'));
const changedFiles = fs.existsSync(changedFilesPath)
	? fs.readFileSync(changedFilesPath, 'utf8').split('\n').filter(Boolean)
	: [];
const missing = new Set();
const patterns = [/\$i18n\.t\(\s*'((?:\\.|[^'\\])*)'/g, /\$i18n\.t\(\s*"((?:\\.|[^"\\])*)"/g];
const translationKeys = new Set(Object.keys(translations));
const pluralKeys = new Set(
	[...translationKeys].map((key) => key.replace(/_(?:zero|one|two|few|many|other)$/, ''))
);

for (const path of changedFiles.filter(
	(path) => /\.(?:js|ts|svelte)$/.test(path) && fs.existsSync(path)
)) {
	const source = fs.readFileSync(path, 'utf8');
	for (const pattern of patterns) {
		for (const match of source.matchAll(pattern)) {
			const key = match[1].replace(/\\(['"\\])/g, '$1');
			if (!translationKeys.has(key) && !pluralKeys.has(key)) missing.add(key);
		}
	}
}

if (missing.size) {
	console.error(`Missing en-US i18n keys:\n${[...missing].sort().join('\n')}`);
	process.exit(1);
}
