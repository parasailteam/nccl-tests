num_gpus = 2
nchannels = 2
num_chunks = num_gpus * nchannels # 96
num_chunks_ring = num_chunks    # 48
def ring_16():
    print('<algo name="test_16_ring8" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(num_chunks, nchannels))
    cnt_chunks = num_chunks_ring / nchannels # 8 = num_gpus/2
    # Ring 16
    for g in range(num_gpus):
        print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="0">'.format(g, num_chunks, num_chunks))
        tbid = -1
        sendpeer = (g+num_gpus-1)%num_gpus
        recvpeer = (g+1)%num_gpus
        for c in range(nchannels):
            tbid = tbid + 1
            print('    <tb id="{}" send="{}" recv="{}" chan="{}">'.format(tbid, sendpeer, recvpeer, c))
            # for s in range(num_gpus-1):
            for s in range(num_gpus):
                src_off = int(((s+g)*nchannels) % num_chunks_ring + c)
                if s == 0:
                    print('      <step s="{}" type="s" srcbuf="o" srcoff="{}" dstbuf="o" dstoff="0" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(s, src_off)) # TODO check
                elif s == num_gpus-1:
                    print('      <step s="{}" type="r" srcbuf="o" srcoff="0" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(s, src_off))
                else:
                    print('      <step s="{}" type="rcs" srcbuf="o" srcoff="0" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(s, src_off))
            print('    </tb>')
        print('  </gpu>')
    print('</algo>')
ring_16()
