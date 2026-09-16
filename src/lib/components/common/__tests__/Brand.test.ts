import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';
import { APP_NAME } from '$lib/constants';

type SvelteNode = {
	type?: string;
	name?: string;
	data?: string;
	value?: unknown;
	expression?: SvelteNode;
	test?: SvelteNode;
	consequent?: SvelteNode;
	alternate?: SvelteNode;
	attributes?: SvelteNode[];
	[key: string]: unknown;
};

const source = readFileSync(resolve(process.cwd(), 'src/lib/components/common/Brand.svelte'), 'utf8');
const ast = parse(source, { modern: true });

const collectNodes = (value: unknown): SvelteNode[] => {
	const nodes: SvelteNode[] = [];

	const visit = (child: unknown) => {
		if (!child || typeof child !== 'object') return;

		if (Array.isArray(child)) {
			child.forEach(visit);
			return;
		}

		const node = child as SvelteNode;
		if (node.type) nodes.push(node);

		for (const [key, nestedChild] of Object.entries(node)) {
			if (!['loc', 'metadata'].includes(key)) visit(nestedChild);
		}
	};

	visit(value);
	return nodes;
};

const nodes = collectNodes(ast.fragment);
const images = nodes.filter((node) => node.type === 'RegularElement' && node.name === 'img');
const attribute = (node: SvelteNode, name: string) =>
	node.attributes?.find((candidate) => candidate.name === name);

describe('application brand', () => {
	it('uses the corporate name by default', () => {
		expect(APP_NAME).toBe('AWG GPT');
		expect(images).toHaveLength(2);
		expect(
			images.map((image) =>
				collectNodes(attribute(image, 'src')).find((node) => node.type === 'Text')?.data
			)
		).toEqual(['/static/awg-logo.svg', '/static/awg-logo-dark.svg']);

		const visibleName = nodes.find(
			(node) => node.type === 'IfBlock' && node.test?.name === 'showName'
		);
		expect(
			collectNodes(visibleName?.consequent).some(
				(node) => node.type === 'ExpressionTag' && node.expression?.name === '$WEBUI_NAME'
			)
		).toBe(true);
	});

	it('shows the configured name through escaped text interpolation', () => {
		const visibleName = nodes.find(
			(node) => node.type === 'IfBlock' && node.test?.name === 'showName'
		);
		const visibleNameNodes = collectNodes(visibleName?.consequent);

		expect(
			visibleNameNodes.some(
				(node) => node.type === 'ExpressionTag' && node.expression?.name === '$WEBUI_NAME'
			)
		).toBe(true);
		expect(visibleNameNodes.some((node) => node.type === 'HtmlTag')).toBe(false);
	});

	it('provides an accessible name when the wordmark is hidden', () => {
		for (const image of images) {
			const altExpression = attribute(image, 'alt')?.value as SvelteNode;
			const condition = altExpression.expression;

			expect(condition?.type).toBe('ConditionalExpression');
			expect(condition?.test?.name).toBe('showName');
			expect(condition?.consequent?.value).toBe('');
			expect(condition?.alternate?.name).toBe('$WEBUI_NAME');
		}
	});
});
