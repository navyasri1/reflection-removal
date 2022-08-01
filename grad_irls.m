function [I_t I_r configs]=grad_irls(I_in, dx,dy,c,h,w,ch)

% %   dx=configs.dx;
% %   dy=configs.dy;
% %   c=configs.c;

configs.dims=[size(I_in,1) size(I_in,2)];
dims=[h w];

%   dims=configs.dims;

  configs.delta=1e-4;
  configs.use_lap=1;
  configs.use_diagnoal=1;
  configs.use_lap2=1;
  configs.niter=20;
  configs.num_px=h*w;

  mk = construct_kernel(h,w,dx,dy,c);
  mh = inv(mk);

  mx = get_fx(h,w);
  my = get_fy(h,w);

  mu = get_fu(h,w);
  mv = get_fv(h,w);

  mlap = get_lap(h,w);

  k=ch;
  I_x=imfilter(I_in, [-1 1]);
  I_y=imfilter(I_in, [-1; 1]);

  out_xi=I_x/2; out_yi=I_y/2;

  out_x=irls_grad(I_x, [], out_xi, mh, configs, mx, my,  mu, mv, mlap);
  outr_x = reshape(mh*(I_x(:)-out_x(:)), dims);
  out_y=irls_grad(I_y, [], out_yi, mh, configs, mx, my, mu, mv, mlap);
  outr_y = reshape(mh*(I_y(:)-out_y(:)), dims);

  I_t=Integration2D(out_x, out_y, I_in);
  I_r=Integration2D(outr_x, outr_y, I_in);
