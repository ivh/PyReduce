sl=dblarr(161)
sp=dblarr(768)
im=dblarr(768,15)
model=im
spold=sp
openr,1,'dump.bin'
readu,1,sl,sp,spold,im,model
close,1

!p.multi=[0,1,3]
plot,sp,xs=1
plot,sl,xs=1
plot,total(im,2)

ss=get_kbrd(1)

shade_surf,im,xs=1,ys=1,ax=70
shade_surf,model,xs=1,ys=1,ax=70
shade_surf,im-model,xs=1,ys=1,ax=70

!p.multi=0
end
