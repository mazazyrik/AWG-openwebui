<script lang="ts">
	import { WEBUI_BASE_URL } from '$lib/constants';
	import { WEBUI_NAME } from '$lib/stores';

	export let className = 'h-10';
	export let showName = false;
</script>

<span class="inline-flex items-center gap-3">
	<img
		src="{WEBUI_BASE_URL}/static/awg-logo.svg"
		alt={showName ? '' : $WEBUI_NAME}
		class="{className} w-auto object-contain dark:hidden"
		draggable="false"
	/>
	<img
		src="{WEBUI_BASE_URL}/static/awg-logo-dark.svg"
		alt={showName ? '' : $WEBUI_NAME}
		class="{className} hidden w-auto object-contain dark:block"
		draggable="false"
	/>
	{#if showName}
		<span class="font-medium tracking-tight text-gray-900 dark:text-gray-100">{$WEBUI_NAME}</span>
	{/if}
</span>
