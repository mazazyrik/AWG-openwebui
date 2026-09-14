<script>
	import Brand from '$lib/components/common/Brand.svelte';
	import { page } from '$app/stores';
</script>

<div class=" bg-white dark:bg-gray-800 min-h-screen">
	<div class="p-8"><Brand showName /></div>
	<div class=" flex h-full">
		<div class="m-auto my-10 dark:text-gray-300 text-3xl font-normal">
			{$page.status}: {$page.error?.message}
		</div>
	</div>
</div>
