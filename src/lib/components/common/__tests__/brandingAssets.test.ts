import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = process.cwd();
const asset = (name: string) => readFileSync(resolve(root, 'static/static', name));
const appHtml = readFileSync(resolve(root, 'src/app.html'), 'utf8');
const rootLayout = readFileSync(resolve(root, 'src/routes/+layout.svelte'), 'utf8');
const manifest: {
	name: string;
	short_name: string;
	start_url: string;
	icons: { src: string; sizes: string }[];
} = JSON.parse(asset('site.webmanifest').toString());

describe('installable application branding', () => {
	it('uses the same name in package and install metadata', () => {
		const pkg = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
		const lock = JSON.parse(readFileSync(resolve(root, 'package-lock.json'), 'utf8'));
		expect(pkg.name).toBe('awg-gpt');
		expect(lock.name).toBe(pkg.name);
		expect(lock.packages[''].name).toBe(pkg.name);
		expect(manifest.name).toBe('AWG GPT');
		expect(manifest.short_name).toBe(manifest.name);
		expect(manifest.start_url).toBe('/');
	});

	it.each(manifest.icons)(
		'ships the declared $sizes install icon',
		(icon: { src: string; sizes: string }) => {
			const png = asset(icon.src.slice('/static/'.length));
			expect(png.subarray(1, 4).toString()).toBe('PNG');
			expect(`${png.readUInt32BE(16)}x${png.readUInt32BE(20)}`).toBe(icon.sizes);
		}
	);

	it.each([
		'favicon.png',
		'favicon.ico',
		'favicon.svg',
		'apple-touch-icon.png',
		'splash.png',
		'splash-dark.png',
		'awg-logo.svg',
		'awg-logo-dark.svg'
	])('keeps the packaged backend fallback identical to %s', (name) => {
		expect(readFileSync(resolve(root, 'backend/open_webui/static', name))).toEqual(asset(name));
	});

	it('ships the same default model icon at the legacy URL', () => {
		expect(readFileSync(resolve(root, 'static/favicon.png'))).toEqual(asset('favicon.png'));
	});

	it('keeps document favicon links on the packaged AWG assets', () => {
		expect(appHtml).toContain('href="/static/favicon.png"');
		expect(appHtml).toContain('href="/static/favicon.svg"');
		expect(appHtml).toContain('href="/static/favicon.ico"');
		expect(rootLayout).toContain(
			'href="{WEBUI_BASE_URL}/static/favicon.png"'
		);
	});
});
