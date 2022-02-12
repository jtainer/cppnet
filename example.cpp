// 
// Testing network.h interface
//
// 2022, Jonathan Tainer
//

#include "network.h"
#include <stdio.h>

int main() {
	
	Network sysnet(1024, 1024, 1024, 1024);
//	for (int i = 0; i < 1; i++)
//		sysnet.randomize(-1.f, 1.f);
	
	DevNetwork devnet(sysnet);
}

