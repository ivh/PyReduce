Function bandsol,aa,rr,DOUBLE=dbl

  n=n_elements(rr)
  sz=size(aa)
  if(n eq 0 or sz(0) ne 2 or sz(1) ne n) then begin
    print,'bandsol solve a sparse system of linear equations with band-diagonal matrix.'
    print,'Band is assumed to be symmetrix relative to the main diaginal. Usage:'
    print,'res=bandsol(a,r[,/DOUBLE])'
    print,'where a is 2D array [n,m] where n - is the number of equations and m'
    print,'        is the width of the band (3 for tri-diagonal system),'
    print,'        m is always an odd number. The main diagonal should be in a(*,m/2)'
    print,'        The first lower subdiagonal should be in a(1:n-1,m-2-1), the first'
    print,'        upper subdiagonal is in a(0:n-2,m/2+1) etc. For example:'
    print,'               / 0 0 X X X \'
    print,'               | 0 X X X X |'
    print,'               | X X X X X |'
    print,'               | X X X X X |'
    print,'           A = | X X X X X |'
    print,'               | X X X X X |'
    print,'               | X X X X X |'
    print,'               | X X X X 0 |'
    print,'               \ X X X 0 0 /'
    print,'      r is the array of RHS of size n.'
    print,'      /DOUBLE forces the calculations to double precision.'
    return,0
  endif
  nd=sz(2)

  if(keyword_set(dbl)) then begin
    a=double(aa)
    r=double(rr)
  endif else begin
    a=float(aa)
    r=double(rr)
  endelse

  for i=0L,n-2L do begin
    r(i)=r(i)/a(i,nd/2) & a(i,*)=a(i,*)/a(i,nd/2)
    for j=1L,(nd/2)<(n-i-1L) do begin
      r(i+j)=r(i+j)-r(i)*a(i+j,nd/2-j)
      a(i+j,0:nd-j-1)=a(i+j,0:nd-j-1)-a(i,j:nd-1)*a(i+j,nd/2-j)
    endfor
  endfor

  r(n-1L)=r(n-1L)/a(n-1L,nd/2)
  for i=n-1L,1L,-1L do begin
    for j=1L,(nd/2)<i do begin
      r(i-j)=r(i-j)-r(i)*a(i-j,nd/2+j)
    endfor
    r(i-1L)=r(i-1L)/a(i-1L,nd/2)
  endfor
  r(0L)=r(0L)/a(0L,nd/2)

  return,r
end

Pro slit_func,im,ycen,sp,sf,OVERSAMPLE=oversample,LAMBDA_SF=lamb_sf $
             ,LAMBDA_SP=lamb_sp,IM_OUT=im_out,USE_COL=use_col $
             ,MASK=mask,NOISE=detector_noise,BAD=jbad,MODEL_ONLY=model $
             ,DEBUG=debug,WING_SMOOTH_FACTOR=wing_smooth_factor $
             ,UNCERTAINTY=unc,PRESET_SLIT_FUNC=preset_slit_func
  common reduce_library,library_name

  if(not keyword_set(library_name)) then begin
    help,calls=a
    delimiter=path_sep()
    i1=strpos(a[0],'<')+1
    prefix=file_dirname(strmid(a[0],i1))
    library_name=prefix+delimiter+'reduce.so.'   $
                                   +!version.os+'.' $
                                   +!version.arch+'.' $
                                   +strtrim(!version.MEMORY_BITS,2)
  endif
  if(keyword_set(detector_noise)) then noise=detector_noise else noise=0
  if(not keyword_set(oversample)) then oversample=1
  if(oversample lt 1) then oversample=1
  if(not keyword_set(mask)) then begin
    mmsk=byte(im*0)+1B
  endif else begin
    if((size(mask))(0) ne (size(im))(0) or $
       (size(mask))(1) ne (size(im))(1) or $
       (size(mask))(2) ne (size(im))(2)) then begin
      print,'SLIT_FUNC: Mask must have the same size as the image'
      stop
    endif
    mmsk=mask
  endelse
  osample=long(oversample)
  oind=lindgen(osample+1L)*(osample+2L)
  weight=1./float(osample)
  if(not keyword_set(lamb_sf)) then lamb_sf=0.1

  for reject=1,1 do begin
    if(keyword_set(use_col)) then begin
      imm=im(use_col,*)
      yycen=ycen(use_col)
      msk=mmsk(use_col,*)
    endif else begin
      use_col=indgen(n_elements(im(*,0)))
      imm=im
      yycen=ycen
      msk=mmsk
    endelse

    sz=size(imm)
    ncol=sz(1)
    nrow=sz(2)
    n=(nrow+1L)*osample+1L

    norm=n_elements(msk)/total(long(msk))

    y=dindgen(n)/float(osample)-1.d0

    bklind=lindgen(osample+1)+n*osample
    olind=oind(0:osample)
    for m=osample+1L,2L*osample do begin
      mm=m-osample
      bklind=[bklind,lindgen(osample+1-mm)+n*m]
      olind=[olind,oind(0:osample-mm)+mm]
    endfor

    if(keyword_set(model)) then begin
      dev=sqrt(mean(sp)+noise*noise)
;      if(keyword_set(noise)) then $
;        dev=sqrt(rebin(sf(osample/2:nrow*osample+osample/2-1),nrow) $
;                *mean(sp)+noise*noise) $
;      else $
;        dev=sqrt(total(msk*(imm-sp#sf)^2)/total(msk))
      goto,model_only
    endif

;    sf=dblarr(nrow)
;    for j=0,nrow-1 do begin
;      sf(j)=median(imm(*,j)*msk(*,j))                   ; Initial guess for the
;    endfor
    if(not keyword_set(preset_slit_func)) then begin     ; Slit function is unknown
      sf=total(imm*msk,1)
      if(osample gt 2 and n_elements(sf) gt 5) then sf=median(sf,5)              ; the spectrum
      if(mean(total(imm,2)) lt 1.d3) then $              ; robust guess for sf
        sf=exp(-((dindgen(nrow)-nrow/2.)/(nrow/4.))^2)   ; in case of low S/N
      sf=sf/total(sf)                                    ; slit function
      sp=total((imm*msk)*(replicate(1.,ncol)#sf),2)*norm ; Initial guess for 
      if(osample gt 2) then sp=median(sp,5)              ; the spectrum
      sp=sp/total(sp)*total(imm*msk)
    endif else begin                                     ; Slit function is preset
      sp=total(imm*msk,2)/(total(msk,2)>1)               ; Initial guess for 
      if(osample gt 2) then sp=median(sp,5)              ; the spectrum
      sp=sp/total(sp)*total(imm*msk)
    endelse

;    outliers=0L

    dev=sqrt(total(msk*(imm-sp#sf)^2)/total(msk))
    j=where(abs(imm-sp#sf) gt 3.*dev,nj)
    if(nj gt 0) then begin
      msk(j)=0B
;      outliers=outliers+nj
;      stop
    endif

;    omega_bar=dblarr(n,nrow+1)
;    AAkl=dblarr(n,2*osample+1)
;    for i=0,ncol-1 do begin
;      yy=y+yycen(i)                            ; Offset SF
;      for j=0l,nrow do begin
;        ind=where(yy ge j and yy lt j+1.d0, nind)     ; Weights are the same within pixel except
;        i1=ind(0) & i2=ind(nind-1)               ; for the first and the last subpixels.
;        bottom=yy(i1)
;        top=1.d0-yy(i2)
;        if(nind gt 2) then o=[bottom,replicate(weight,i2-i1-1),top] $
;        else if(nind eq 2) then o=[bottom,weight] $
;        else if(nind eq 1) then o=bottom
;        if(nind gt 0) then omega_bar(i1:i2,j)=o ; Weight for a given row
;      endfor
;      for j=0l,nrow do begin
;      end
;    endfor
;stop
;    AAkl=(sp^2)#(omega_bar#transpose(omega_bar))

    iter=0
next:
    iter=iter+1                                ; Iteration counter

    if(not keyword_set(preset_slit_func)) then begin ; Solve for the slit function
      Akl=dblarr(n,2*osample+1)                  ; Initialize matrix
      Bl=dblarr(n)                               ; and RHS
      omega=replicate(weight,osample+1L)         ; Replicate constant weights

      if(keyword_set(debug)) then time0=systime(1)
      for i=0,ncol-1 do begin                    ; Fill up matrix and RHS
        yy=y+yycen(i)                            ; Offset SF

        ind=where(yy ge 0 and yy lt 1, nind)     ; Weights are the same within pixel except
        i1=ind(0) & i2=ind(nind-1)               ; for the first and the last subpixels.
        omega(0)=yy(i1)                          ; Fix the first and the last subpixel, here
        omega(osample)=1.-yy(i2)                 ; the weight is split between the two subpixels
        bkl=dblarr(n,2*osample+1)                ; Band-diagonal part that will contain omega#omega
        o=omega#omega & o(osample,osample)=o(osample,osample)+o(0,0)
        oo=o(olind)
        for l=0,nrow-1 do begin
;         for m=osample,2L*osample do begin     ; Explicit and slow way of filling bkl
;           mm=m-osample
;           bkl(l*osample+i1:l*osample+i1+osample-mm,m)=o(oind(0:osample-mm)+mm)
;         endfor
          bkl(l*osample+i1+bklind)=oo*msk(i,l)
        endfor
        oo=o(osample,osample)
        for l=1,nrow-1 do bkl(l*osample+i1,osample)=oo*msk(i,l)
        bkl(nrow*osample+i1,osample)=omega(osample)^2*msk(i,nrow-1)
        for m=0L,osample-1L do bkl(osample-m:n-1L,m)=bkl(0L:n-1L-osample+m,2L*osample-m)
        Akl=Akl+sp(i)^2*bkl

        o=dblarr(n)
        for l=0,nrow-1 do o(l*osample+i1:l*osample+i1+osample)= $
                                          imm(i,l)*weight*msk(i,l)
        for l=1,nrow-1 do o(l*osample+i1)=imm(i,l-1)*omega(osample)*msk(i,l-1) $
                                         +imm(i,l)*omega(0)*msk(i,l)
        o(i1)=imm(i,0)*omega(0)*msk(i,0)
        o(nrow*osample+i1)=imm(i,nrow-1)*omega(osample)*msk(i,nrow-1)
        Bl=Bl+sp(i)*o
      endfor
      if(keyword_set(debug)) then time1=systime(1)
;stop
      lambda = lamb_sf*total(Akl(*,osample))/n
      if(keyword_set(wing_smooth_factor) and iter gt 1) then begin
;        lambda=lambda*(1.+wing_smooth_factor*(2.d0*dindgen(n)/(n-1)-1.d0)^2)
        lambda=lambda*(1.+wing_smooth_factor/(sf>1.d-5))
      endif else begin
        lambda=replicate(lambda,n)
      endelse

; 1st order Tikhonov regularization (minimum 1st derivatives)
; Add the following 3-diagonal matrix * lambda:
;  1 -1  0  0  0  0
; -1  2 -1  0  0  0
;  0 -1  2 -1  0  0
;  0  0 -1  2 -1  0
;      .  .  .

;      Akl(  0,osample)=Akl(  0,osample)+lambda   ; +lambda to the upper-left element
;      Akl(n-1,osample)=Akl(n-1,osample)+lambda   ; and to the lower-right
;      Akl(1L:n-2L,osample)=Akl(1L:n-2L,osample)+2.*lambda    ; +2*lambda to the rest of the main diagonal
;      Akl(0L:n-2L,osample+1L)=Akl(0L:n-2L,osample+1L)-lambda ; -lambda to the upper sub-diagonal
;      Akl(1L:n-1L,osample-1L)=Akl(1L:n-1L,osample-1L)-lambda ; -lambda to the lower sub-diagonal

      Akl(  0,osample)=Akl(  0,osample)+lambda[   0]; +lambda to the upper-left element
      Akl(n-1,osample)=Akl(n-1,osample)+lambda[n-1L]; and to the lower-right
      Akl(1L:n-2L,osample)=Akl(1L:n-2L,osample)+2.*lambda[1L:n-2L]; +2*lambda to the rest of the main diagonal
      Akl(0L:n-2L,osample+1L)=Akl(0L:n-2L,osample+1L)-lambda[0L:n-2L]; -lambda to the upper sub-diagonal
      Akl(1L:n-1L,osample-1L)=Akl(1L:n-1L,osample-1L)-lambda[1L:n-1L]; -lambda to the lower sub-diagonal
;
; 2nd order Tikhonov regularization (minimum 2nd derivative)
; Add the following 5-diagonal matrix * lambda:
;  1 -2  1  0  0  0
; -2  5 -4  1  0  0
;  1 -4  6 -4  1  0
;  0  1 -4  6 -4  1
;      .  .  .

;      lambda=0.1*lambda
;      Akl(  0,osample)=Akl(  0,osample)+1.*lambda ; Main diagonal
;      Akl(n-1,osample)=Akl(n-1,osample)+1.*lambda
;      Akl(  1,osample)=Akl(  1,osample)+5.*lambda
;      Akl(n-2,osample)=Akl(n-2,osample)+5.*lambda
;      Akl(2L:n-3L,osample)=Akl(2L:n-3L,osample)+6.*lambda
;      Akl(0L,osample+1L)=Akl(0L,osample+1L)-2.*lambda ; upper sub-diagonal
;      Akl(n-2L,osample+1L)=Akl(n-2L,osample+1L)-2.*lambda
;      Akl(1L:n-3L,osample+1L)=Akl(1L:n-3L,osample+1L)-4.*lambda
;      Akl(1L,osample-1L)=Akl(1L,osample-1L)-2.*lambda ; lower sub-diagonal
;      Akl(n-1L,osample-1L)=Akl(n-1L,osample-1L)-2.*lambda
;      Akl(2L:n-2L,osample-1L)=Akl(2L:n-2L,osample-1L)-4.*lambda
;
;      sf=bandsol(Akl,Bl,/DOUBLE)
;      sf=sf/total(sf)*osample

      i=CALL_EXTERNAL(library_name, 'bandsol', $
                      Akl, Bl, n, 2*osample+1L)
      sf=Bl>0
      sf=sf/total(sf)*osample

      if(keyword_set(debug)) then time2=systime(1)
    endif

    sp_old=sp
    r=sp

    omega=replicate(weight,osample)

    if(keyword_set(debug)) then sssf=dblarr(nrow,ncol)
;    sssf=dblarr(ncol,nrow)

    dev_new=0.d0
    for i=0,ncol-1 do begin                    ; Evaluate the new spectrum
      yy=y+yycen(i)                            ; Offset SF
;      omega=fltarr(n,nrow)                     ; weights for ovsersampling
                                               ; omega(k,j) is how much
                                               ; point k in oversampled SF
                                               ; contributes to the image pixel j
;      for j=0,nrow-1 do begin
;        ind=where(yy gt j and yy lt j+1, nind)
;        if(nind gt 0) then begin
;          omega(ind,j)=weight                 ; All sub-pixels that fall to image
;          i1=ind(0)                           ; pixel j have weight 1/osample
;          i2=ind(nind-1)
;          omega(i1,j)=yy(i1)-j                ; On the two ends we may have
;          omega(i2+1,j)=j+1-yy(i2)            ; boundary crossing so weights
;                                              ; could be less than 1./osample
;        endif
;      endfor
;      omega=reform(sf#omega)
      i1=where(yy ge 0 and yy lt nrow, nind)
      i2=i1(nind-1)
      i1=i1(0)
      omega(0)=yy(i1)
      ssf=reform(sf(i1:i2),osample,nrow)
      o=reform(ssf##omega)
      yyy=nrow-yy(i2)
      o(0:nrow-2)=o(0:nrow-2)+reform(ssf(0,1:nrow-1))*yyy
      o(nrow-1)=o(nrow-1)+sf(i2+1)*yyy

      r(i) =((imm(i,*)*msk(i,*))#o)
      sp(i)=total(o^2*msk(i,*))
      if(sp(i) eq 0.d0) then sp(i)=total(o^2)

      if(keyword_set(debug)) then sssf[*,i]=r[i]/sp[i]*o

;Locate and mask outliers
      if(iter gt 1) then begin
        norm=(r[i]/sp[i])
        j=where(abs((imm[i,*]-norm*o)) gt 6.*dev,nj,COMPLEMENT=b)
;        j=where(abs((imm[i,*]/norm-o)) gt 7.*dev,nj,COMPLEMENT=b)
;if(iter eq 5 and i ge ncol/3) then stop
        if(nj gt 0) then begin
          msk(i,j)=0B
        endif
        if(nj lt nrow) then msk[i,b]=1B*(mmsk[use_col,*])[i,b]
        dev_new=dev_new+total(msk[i,*]*(imm[i,*]-norm*o)^2)
;        dev_new=dev_new+total(msk[i,*]*(imm[i,*]/norm-o)^2)
      endif
    endfor

    if(iter gt 1) then dev=sqrt(noise*noise+dev_new/total(msk))
;    if(iter gt 1) then dev=sqrt(dev_new/total(msk))

    if(keyword_set(lamb_sp)) then begin
      lambda = lamb_sp*total(sp)/ncol
      a =[0.,replicate(-lambda,ncol-1)]
      b =[lambda+1.,replicate(2.*lambda+1.,ncol-2),lambda+1.]
      c =[replicate(-lambda,ncol-1),0.]
      sp=trisol(a,b,c,r/sp)
    endif else sp=r/sp

;    yy  =imm/(sp#replicate(1.d,nrow))
;    dev_new=total(msk*(yy-sssf)^2)
    if(iter gt 1) then dev=sqrt(dev_new/total(msk))

    if(keyword_set(debug)) then begin
      time3=systime(1)
      !p.multi=[0,1,2]
      x=dindgen(n_elements(im(0,*)))+0.5d0+0.5d0/osample

      sssf=sssf/(replicate(1.d,nrow)#sp)
;      sssf=transpose(sssf)
      xx=x-ycen(0)
      yy=reform(im(0,*)/sp(0))
      for i=1,ncol-1 do begin
        xx=[xx,x-ycen(i)]
        yy=[yy,reform(im(i,*)/sp(i))]
      endfor
      ii=(reverse(sort(yy)))(0:10)
      print, ii/nrow, ii mod nrow
      ii=sort(xx)
      xx=xx(ii)
      yy=yy(ii)
      yr=minmax(yy)
      plot,xx,sssf(ii),xs=3,ys=1,yr=yr,title='Slit Function. Iteration='+strtrim(iter,2)
      oplot,!x.crange,mean(sf)+[0,0],line=3
      oplot,xx,yy,psym=3
      bad=where((transpose(msk))[ii] eq 0b, nbad)
      if(nbad gt 0) then oplot,xx[bad],yy[bad],psym=1,col=c24(3)
      oplot,xx,yy-sssf(ii)+mean(sf),psym=3,col=c24(2)
      oplot,xx,sssf(ii),col=c24(3)
      plot,sp,xs=3,ys=3,title='Spectrum'
;if(iter eq 8) then stop
;      wshow,0
      !p.multi=0
      print,'Maximum change in the spectrum is:',max(abs(sp-sp_old)/max(sp)),dev $
           ,sqrt(total((yy-yyy)^2)/n_elements(yy))
      print,time1-time0,time2-time1,time3-time2
      answ=get_kbrd(1)
    endif

;    if(iter lt 4 and (max(abs(sp-sp_old)/max(sp)) gt 1.d-4 or outliers gt 0L)) then goto,next
    if(iter lt 18 and max(abs(sp-sp_old)/max(sp)) gt 1.d-5) then goto,next
;      stop

model_only:
    jbad=0L
    if(arg_present(im_out)) then begin
      ncol=n_elements(ycen)
      if(keyword_set(use_col)) then sp=interpol(sp,use_col,findgen(ncol))
      im_out=fltarr(ncol,nrow)
      unc=fltarr(ncol)
      omega=replicate(weight,osample)
      for i=0,ncol-1 do begin                    ; Evaluate the new spectrum
        yy=y+ycen(i)                             ; Offset SF
;        omega=fltarr(n,nrow)                     ; weights for ovsersampling
;                                                 ; omega(k,j) is how much
;                                                 ; point k in oversampled SF
;                                                 ; contributes to the image pixel j
;        for j=0,nrow-1 do begin
;          ind=where(yy gt j and yy lt j+1, nind)
;          if(nind gt 0) then begin
;            omega(ind,j)=weight                  ; All sub-pixels that fall to image
;            i1=ind(0)                            ; pixel j have weight 1/osample
;            i2=ind(nind-1)
;            omega(i1,j)=yy(i1)-j                 ; On the two ends we may have
;            omega(i2+1,j)=j+1-yy(i2)             ; boundary crossing so weights
;                                               ; could be less than 1./osample
;          endif
;        endfor
;        omega=sf#omega
        i1=where(yy ge 0 and yy lt nrow, nind)
        i2=i1(nind-1)
        i1=i1(0)
        omega(0)=yy(i1)
        ssf=reform(sf(i1:i2),osample,nrow)
        o=reform(reform(ssf##omega))
        yyy=nrow-yy(i2)
        o(0:nrow-2)=o(0:nrow-2)+reform(ssf(0,1:nrow-1))*yyy
        o(nrow-1)=o(nrow-1)+sf(i2+1)*yyy

        j=where(abs(im(i,*)-sp(i)*o) lt 5*dev,nj,COMPLEMENT=b)

;        if(nj gt 0) then $             ; Good pixels in column i
;          sp(i)=total(im(i,j))/total(o(j))

        if(nj lt nrow) then $          ; Bad pixels in column i
          jbad=[jbad,long(nrow)*i+b]

        if(nj gt 2) then begin
          ss=total((im(i,j)-sp(i)*o(j))^2)
          xx=total((o(j)-mean(o(j)))^2)*(nj-2)
          unc(i)=ss/xx
        endif else unc(i)=0.
        im_out(i,*)=sp(i)*o
      endfor
;      stop
    endif
  endfor
  if(n_elements(jbad) gt 1) then jbad=jbad(1:n_elements(jbad)-1) $
  else                           jbad=-1

;  j=where(total(msk,2) eq 0, nj, COMPLEMENT=b)
;  if(nj gt 0) then sp(j)=interpol(sp(b),float(b),float(j))

  return
end

;restore,'slit_func.sav'
;jgood=0
;slit_func,sf,yc,sp,sfsm,OVERSAMPLE=10,IM_OUT=sfbin,LAMBDA_SF=10. $
;            ,USE_COL=jgood,BAD=jbad,MASK=msk
;end
