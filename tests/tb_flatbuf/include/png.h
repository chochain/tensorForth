// ─── PNG Encoder (uncompressed deflate stored blocks) ────────────────────────
namespace png {
    
static uint32_t adler32(const uint8_t* d, size_t n) {
    uint32_t s1=1,s2=0; for(size_t i=0;i<n;i++){s1=(s1+d[i])%65521;s2=(s2+s1)%65521;}
    return (s2<<16)|s1;
}
static uint32_t crc32_png(const uint8_t* d, size_t n) {
    static uint32_t t[256]; static bool init=false;
    if(!init){for(uint32_t i=0;i<256;i++){uint32_t c=i;for(int k=0;k<8;k++)c=(c&1)?(0xEDB88320u^(c>>1)):(c>>1);t[i]=c;}init=true;}
    uint32_t c=0xFFFFFFFFu; for(size_t i=0;i<n;i++)c=t[(c^d[i])&0xFF]^(c>>8); return c^0xFFFFFFFFu;
}
static void push_be32(std::vector<uint8_t>& v,uint32_t n){
    v.push_back((n>>24)&0xFF);v.push_back((n>>16)&0xFF);v.push_back((n>>8)&0xFF);v.push_back(n&0xFF);}
static void write_chunk(std::vector<uint8_t>& out,const char type[4],const std::vector<uint8_t>& data){
    push_be32(out,static_cast<uint32_t>(data.size()));
    out.insert(out.end(),type,type+4); out.insert(out.end(),data.begin(),data.end());
    std::vector<uint8_t> ci; ci.insert(ci.end(),type,type+4); ci.insert(ci.end(),data.begin(),data.end());
    push_be32(out,crc32_png(ci.data(),ci.size()));
}
inline std::vector<uint8_t> raw2png(int w,int h,const std::vector<uint8_t>& px,int ch=3){
    std::vector<uint8_t> out;
    static const uint8_t sig[]={137,80,78,71,13,10,26,10}; out.insert(out.end(),sig,sig+8);
    {std::vector<uint8_t> ihdr; push_be32(ihdr,w); push_be32(ihdr,h);
     ihdr.push_back(8); ihdr.push_back(ch==1?0:(ch==4?6:2));
     ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0); write_chunk(out,"IHDR",ihdr);}
    int rb=w*ch; std::vector<uint8_t> sl;
    for(int y=0;y<h;y++){sl.push_back(0);sl.insert(sl.end(),px.begin()+y*rb,px.begin()+(y+1)*rb);}
    std::vector<uint8_t> zlib; zlib.push_back(0x78); zlib.push_back(0x01);
    size_t pos=0,tot=sl.size();
    while(pos<tot){size_t bsz=std::min(tot-pos,(size_t)65535);bool last=(pos+bsz>=tot);
        uint16_t bl=static_cast<uint16_t>(bsz),bn=static_cast<uint16_t>(~bl);
        zlib.push_back(last?0x01:0x00);
        zlib.push_back(bl&0xFF);zlib.push_back((bl>>8)&0xFF);
        zlib.push_back(bn&0xFF);zlib.push_back((bn>>8)&0xFF);
        zlib.insert(zlib.end(),sl.begin()+pos,sl.begin()+pos+bsz);pos+=bsz;}
    push_be32(zlib,adler32(sl.data(),sl.size()));
    write_chunk(out,"IDAT",zlib); write_chunk(out,"IEND",{}); return out;
}

} // namespace png

