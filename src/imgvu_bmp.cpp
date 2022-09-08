#include <stdio.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable : 4996)  // disable deprecated warning
#endif

#pragma pack(1)

typedef struct { unsigned char x, y, z, w; } uchar4;
typedef struct {
    short type;
    int   size;
    short reserved1;
    short reserved2;
    int   offset;
} BMPHeader;

typedef struct {
    int      size;
    int      width;
    int      height;
    short    planes;
    short    bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int      xPelsPerMeter;
    int      yPelsPerMeter;
    int      clrUsed;
    int      clrImportant;
} BMPInfoHeader;
///
/// load BMP file
///
extern "C" void load_bmp(
    const char *fname, uchar4 **bmp, int *H, int *W) {
    BMPHeader     hdr;
    BMPInfoHeader info;
    FILE *fd;

    printf("Loading %s...\n", fname);
    if (sizeof(uchar4) != 4) {
        printf("***Bad uchar4 size***\n");
        exit(EXIT_SUCCESS);
    }
    if (!(fd = fopen(fname, "rb"))) {
        printf("***BMP load error: file access denied***\n");
        exit(EXIT_SUCCESS);
    }
    
    fread(&hdr, sizeof(hdr), 1, fd);
    if (hdr.type != 0x4D42) {
        printf("***BMP load error: bad file format***\n");
        exit(EXIT_SUCCESS);
    }
    
    fread(&info, sizeof(info), 1, fd);
    if (info.bitsPerPixel != 24) {
        printf("***BMP load error: invalid color depth***\n");
        exit(EXIT_SUCCESS);
    }
    if (info.compression) {
        printf("***BMP load error: compressed image***\n");
        exit(EXIT_SUCCESS);
    }

    *W   = info.width;
    *H   = info.height;
    *bmp = (uchar4*)malloc(info.width * info.height * 4);

    printf("BMP width: %u\n",  info.width);
    printf("BMP height: %u\n", info.height);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(info), SEEK_CUR);

    uchar4 *p = *bmp;
    int z = (4 - (3 * info.width) % 4) % 4;
    for (int y = 0; y < info.height; y++) {
        for (int x = 0; x < info.width; x++, p++) {
            p->z = fgetc(fd);
            p->y = fgetc(fd);
            p->x = fgetc(fd);
        }
        for (int x = 0; x < z; x++) fgetc(fd);  // skip padding
    }
    if (ferror(fd)) {
        printf("***Unknown BMP load error.***\n");
        free(bmp);
        exit(EXIT_SUCCESS);
    }
    else printf("BMP file loaded successfully!\n");

    fclose(fd);
}

