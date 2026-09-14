import { afterEach, describe, expect, it } from 'vitest';
import { render } from 'svelte/server';
import { WEBUI_NAME } from '$lib/stores';
import { APP_NAME } from '$lib/constants';
import Brand from '../Brand.svelte';

afterEach(() => WEBUI_NAME.set(APP_NAME));

describe('application brand', () => {
	it('uses the corporate name by default', () => {
		expect(APP_NAME).toBe('AWG GPT');
		const { body } = render(Brand, { props: { showName: true } });
		expect(body).toContain('AWG GPT');
		expect(body).toContain('/static/awg-logo.svg');
		expect(body).toContain('/static/awg-logo-dark.svg');
	});

	it('shows the configured name and escapes it as text', () => {
		WEBUI_NAME.set('<script>custom</script>');
		const { body } = render(Brand, { props: { showName: true } });
		expect(body).toContain('&lt;script>custom&lt;/script>');
		expect(body).not.toContain('<script>custom</script>');
	});

	it('provides an accessible name when the wordmark is hidden', () => {
		WEBUI_NAME.set('AWG GPT QA');
		const { body } = render(Brand);
		expect(body).toContain('alt="AWG GPT QA"');
	});
});
