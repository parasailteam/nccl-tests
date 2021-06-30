num_gpus = 16
nchannels = 12
num_chunks = num_gpus * nchannels # 96
num_chunks_ring = num_chunks    # 48
def ring_16():
    print('<algo name="test_{}_ring8" nchunksperloop="{}" nchannels="{}" redop="sum" proto="Simple">'.format(num_gpus, num_chunks, nchannels))
    cnt_chunks = num_chunks_ring / nchannels # 8 = num_gpus/2
    # Ring 16
    for g in range(num_gpus):
        print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="0">'.format(g, num_chunks, num_chunks//num_gpus)) #o_chunks is 1 because this is not inplace
        tbid = -1
        recvpeer =  (g+num_gpus-1)%num_gpus
        sendpeer = (g+1)%num_gpus
        ring = [-1 for g in range(num_gpus)]
        ring[0] = g
        ringidx = 0
        ringval = g
        while ringidx < len(ring):
            ring[ringidx] = ringval
            ringval = (ringval + 1) % num_gpus
            ringidx += 1
        # (recvpeer, sendpeer) = (sendpeer, recvpeer)
        for c in range(nchannels):
            tbid = tbid + 1
            print('    <tb id="{}" send="{}" recv="{}" chan="{}">'.format(tbid, sendpeer, recvpeer, c))
            for s in range(num_gpus):
                src_off = ((ring[num_gpus - s - 1]) % num_gpus)*nchannels + c #int(((s+g)*nchannels) % num_chunks_ring + c)
                dst_off = c
                if s == 0:
                    print('      <step s="{}" type="s" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="0" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(s, src_off)) # TODO check
                elif s == num_gpus-1:
                    print('      <step s="{}" type="rrc" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(s, src_off, dst_off))
                else:
                    print('      <step s="{}" type="rrs" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="0" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(s, src_off))
            print('    </tb>')
        print('  </gpu>')
    print('</algo>')
ring_16()
