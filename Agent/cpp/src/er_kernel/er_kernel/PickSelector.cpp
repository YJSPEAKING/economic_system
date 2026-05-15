#include "stdafx.h"
#include "PickSelector.h"
#include <sys/timeb.h>

static bool g_use_fixed_seed = false;

inline UINT32 make_time_seed() {
	struct timeb T;
	ftime(&T);
#pragma warning(push)
#pragma warning(disable: 4244)
	UINT32 t = T.time * 1000 + T.millitm;
#pragma warning(pop)
	return t;
}

void set_er_rand_seed(INT32 seed) {
	if (seed >= 0) {
		g_use_fixed_seed = true;
		srand((UINT32)seed);
	}
	else {
		g_use_fixed_seed = false;
		srand(make_time_seed());
	}
}

void init_er_rand_seed() {
	if (g_use_fixed_seed) return;
	UINT32 t = make_time_seed();
	srand(t);
}

inline UINT64 rand_UINT64(UINT64 ceil = 0xffffffffffffffffULL) {
	UINT64 R = 0;
	UINT64 C = ceil;
	while (C) {
		R <<= 15;
		R |= ((UINT64)rand()) & 0x7fff;
		C >>= 15;
	}
	return R % ceil;
}

PS_UniRand::PS_UniRand() {
	pick_num = 0;
	init_er_rand_seed();
}

void PS_UniRand::select(BATCH_SIZE_T batch_size, LH_REC_T* hPICKs) {
	LH_REC_T stripe_size = (pick_num / batch_size + 1) * 2;
	LH_REC_T* iter = hPICKs;
	LH_REC_T r = 0;
	for (BATCH_SIZE_T i = 0; i < batch_size; ++i, ++iter) {
		r = (r + rand_UINT64(stripe_size)) % pick_num;
		*iter = r;
	}
}

char* PS_UniRand::serialize(char* pSer) {
	return WRITE_BYTES(pSer, pick_num, LH_REC_T);
}

char* PS_UniRand::unserialize(char* pSer) {
	return READ_BYTES(pSer, pick_num, LH_REC_T);
}
