#include "distance.h"

int convert_pixel_to_distance(int pixel){
	if(pixel > 243) return 5;
	else if(pixel >= 238) return 10;
	else if(pixel >= 226) return 11;
	else if(pixel >= 214) return 12;
	else if(pixel >= 202) return 13;
	else if(pixel >= 190) return 14;
	else if(pixel >= 178) return 15;
	else if(pixel >= 166) return 16;
	else if(pixel >= 154) return 17;
	else if(pixel >= 142) return 18;
	else if(pixel >= 130) return 19;
	else if(pixel >= 120) return 20;
	else if(pixel >= 116) return 21;
	else if(pixel >= 112) return 22;
	else if(pixel >= 108) return 23;
	else if(pixel >= 104) return 24;
	else if(pixel >= 100) return 25;
	else if(pixel >= 96) return 26;
	else if(pixel >= 91) return 27;
	else if(pixel >= 87) return 28;
	else if(pixel >= 83) return 29;
	else if(pixel >= 80) return 30;
	else if(pixel >= 78) return 31;
	else if(pixel >= 76) return 32;
	else if(pixel >= 74) return 33;
	else if(pixel >= 72) return 34;
	else if(pixel >= 69) return 35;
	else if(pixel >= 67) return 36;
	else if(pixel >= 65) return 37;
	else if(pixel >= 63) return 38;
	else if(pixel >= 61) return 39;
	else if(pixel >= 59) return 40;
	else if(pixel >= 58) return 41;
	else if(pixel >= 57) return 42;
	else if(pixel >= 56) return 43;
	else if(pixel >= 54) return 44;
	else if(pixel >= 53) return 45;
	else if(pixel >= 52) return 46;
	else if(pixel >= 51) return 47;
	else if(pixel >= 50) return 48;
	else if(pixel >= 49) return 49;
	else if(pixel >= 48) return 50;
	else if(pixel >= 47) return 51;
	else if(pixel >= 46) return 52;
	else if(pixel >= 45) return 53;
	else if(pixel >= 44) return 54;
	else if(pixel >= 43) return 55;
	else if(pixel >= 42) return 56;
	else if(pixel >= 41) return 57;
	else if(pixel >= 40) return 59;
	else if(pixel >= 39) return 61;
	else if(pixel >= 38) return 63;
	else if(pixel >= 37) return 65;
	else if(pixel >= 36) return 67;
	else if(pixel >= 35) return 69;
	else if(pixel >= 34) return 71;
	else if(pixel >= 33) return 73;
	else if(pixel >= 32) return 75;
	else if(pixel >= 31) return 79;
	else if(pixel >= 30) return 85;
	else if(pixel >= 29) return 90;
	else if(pixel >= 28) return 100;
	else return 100;
}
