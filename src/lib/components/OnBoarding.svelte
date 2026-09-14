<script>
	import { getContext } from 'svelte';
	import { WEBUI_NAME } from '$lib/stores';
	import Brand from '$lib/components/common/Brand.svelte';

	const i18n = getContext('i18n');

	export let show = true;
	export let getStartedHandler = () => {};
</script>

{#if show}
	<div
		class="awg-welcome relative min-h-screen w-full overflow-hidden text-gray-900 dark:text-gray-100"
	>
		<div class="absolute top-6 left-6 sm:top-10 sm:left-10">
			<Brand showName />
		</div>

		<div class="mx-auto flex min-h-screen max-w-6xl items-center px-6 py-32 sm:px-10">
			<div class="max-w-2xl">
				<div class="mb-10"><Brand className="h-24 sm:h-32" /></div>
				<div class="mb-4 text-sm font-medium tracking-widest text-brand">{$WEBUI_NAME}</div>
				<h1 class="text-4xl leading-tight font-medium tracking-tight sm:text-5xl">
					{$i18n.t('Welcome to your AI home.')}
				</h1>
				<p class="mt-6 max-w-xl text-base leading-relaxed text-gray-600 dark:text-gray-400">
					{$i18n.t('How can I help you today?')}
				</p>
				<button
					class="mt-10 rounded-xl bg-brand px-8 py-3 text-sm font-medium text-brand-foreground transition hover:opacity-90"
					on:click={getStartedHandler}
				>
					{$i18n.t('Get started')}
				</button>
			</div>
		</div>
	</div>
{/if}
