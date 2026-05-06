#include "stdafx.h"
#include "PickSelector.h"
#include <sys/timeb.h>

inline void set_rand_seed() {
    // ==========================================
    // 🚀 【修改】废弃时间戳种子，强制使用固定种子
    // 彻底消灭 C++ 底层采样带来的毫秒级随机性
    // ==========================================
    srand(184); // 直接硬编码为你的 config.random_seed
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
	set_rand_seed();
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
