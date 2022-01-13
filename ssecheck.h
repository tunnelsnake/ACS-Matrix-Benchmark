int sse_enabled() {
	return __builtin_cpu_supports("sse") > 0 ? 1 : 0;
}

int avx_enabled() {
	return __builtin_cpu_supports("avx") > 0 ? 1 : 0;
}

int avx2_enabled() {
	return __builtin_cpu_supports("avx2") > 0 ? 1 : 0;
}