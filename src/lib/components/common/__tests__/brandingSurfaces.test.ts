import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';

type SvelteNode = {
	type?: string;
	name?: string;
	start?: number;
	end?: number;
	attributes?: { name?: string }[];
	[key: string]: unknown;
};

const root = process.cwd();

const componentTree = (path: string) => {
	const source = readFileSync(resolve(root, path), 'utf8');
	const ast = parse(source, { modern: true });
	const nodes: SvelteNode[] = [];

	const visit = (value: unknown) => {
		if (!value || typeof value !== 'object') return;

		if (Array.isArray(value)) {
			value.forEach(visit);
			return;
		}

		const node = value as SvelteNode;
		if (node.type) nodes.push(node);

		for (const [key, child] of Object.entries(node)) {
			if (!['loc', 'metadata'].includes(key)) visit(child);
		}
	};

	visit(ast.fragment);

	return { nodes, source };
};

const componentsNamed = (nodes: SvelteNode[], name: string) =>
	nodes.filter((node) => node.type === 'Component' && node.name === name);

describe('branded application surfaces', () => {
	it('keeps the new-chat placeholder to an unnamed brand and message input', () => {
		const { nodes } = componentTree('src/lib/components/chat/Placeholder.svelte');
		const brands = componentsNamed(nodes, 'Brand');

		expect(brands).toHaveLength(1);
		expect(brands[0].attributes?.map((attribute) => attribute.name)).not.toContain('showName');
		expect(componentsNamed(nodes, 'MessageInput')).toHaveLength(1);
		expect(nodes.some((node) => node.type === 'RegularElement' && node.name === 'img')).toBe(false);
		expect(nodes.some((node) => node.type === 'Identifier' && node.name === 'selectedModelName')).toBe(
			false
		);
		expect(
			nodes.some((node) => node.type === 'Identifier' && node.name === 'selectedModelDescription')
		).toBe(false);
	});

	it('uses the AWG Brand component on app and responsive sidebar variants', () => {
		const appSidebar = componentTree('src/lib/components/app/AppSidebar.svelte');
		const sidebar = componentTree('src/lib/components/layout/Sidebar.svelte');

		expect(componentsNamed(appSidebar.nodes, 'Brand')).toHaveLength(2);
		expect(componentsNamed(sidebar.nodes, 'Brand')).toHaveLength(2);
	});

	it('keeps the AWG brand and favicon fallback on the chat placeholder', () => {
		const { nodes, source } = componentTree('src/lib/components/chat/ChatPlaceholder.svelte');
		const fallbackPaths = nodes
			.filter((node) => node.type === 'TemplateElement')
			.map((node) => source.slice(node.start, node.end));

		expect(componentsNamed(nodes, 'Brand')).toHaveLength(1);
		expect(fallbackPaths).toContain('/static/favicon.png');
		expect(fallbackPaths).not.toContain('/static/logo.png');
	});
});
