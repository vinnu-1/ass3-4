<!DOCTYPE html>
<!-- saved from url=(0056)https://colab.research.google.com/#scrollTo=r69rv1NFa3-w -->
<html lang="en" theme="adaptive" editor="Default Dark" style="--colab-code-font-family: monospace;"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><meta http-equiv="origin-trial" content="Az520Inasey3TAyqLyojQa8MnmCALSEU29yQFW8dePZ7xQTvSt73pHazLFTK5f7SyLUJSo2uKLesEtEa9aUYcgMAAACPeyJvcmlnaW4iOiJodHRwczovL2dvb2dsZS5jb206NDQzIiwiZmVhdHVyZSI6IkRpc2FibGVUaGlyZFBhcnR5U3RvcmFnZVBhcnRpdGlvbmluZyIsImV4cGlyeSI6MTcyNTQwNzk5OSwiaXNTdWJkb21haW4iOnRydWUsImlzVGhpcmRQYXJ0eSI6dHJ1ZX0="><meta http-equiv="origin-trial" content="Az520Inasey3TAyqLyojQa8MnmCALSEU29yQFW8dePZ7xQTvSt73pHazLFTK5f7SyLUJSo2uKLesEtEa9aUYcgMAAACPeyJvcmlnaW4iOiJodHRwczovL2dvb2dsZS5jb206NDQzIiwiZmVhdHVyZSI6IkRpc2FibGVUaGlyZFBhcnR5U3RvcmFnZVBhcnRpdGlvbmluZyIsImV4cGlyeSI6MTcyNTQwNzk5OSwiaXNTdWJkb21haW4iOnRydWUsImlzVGhpcmRQYXJ0eSI6dHJ1ZX0="><style>body {transition: opacity ease-in 0.2s; } 
body[unresolved] {opacity: 0; display: block; overflow: hidden; position: relative; } 
</style><script type="text/javascript" async="" src="./untitled10_files/recaptcha__en.js.download" crossorigin="anonymous" integrity="sha384-0lJkOVHDy3ItYlCbUoEzThjP3hLhLYfEFPAkVOCxnJpb5K9Fllso+S8TRBILcfPo" nonce=""></script><script type="text/javascript" async="" src="./untitled10_files/recaptcha__en.js.download" crossorigin="anonymous" integrity="sha384-0lJkOVHDy3ItYlCbUoEzThjP3hLhLYfEFPAkVOCxnJpb5K9Fllso+S8TRBILcfPo" nonce=""></script><script src="./untitled10_files/cb=gapi.loaded_1" nonce="" async=""></script><script src="./untitled10_files/cb=gapi.loaded_0" nonce="" async=""></script><script type="text/javascript" async="" src="./untitled10_files/js" nonce=""></script><script async="" src="./untitled10_files/analytics.js.download"></script><script nonce="">
      document.addEventListener('keydown', (e) => {
        // Stop propagation on ESC because otherwise it will halt outbound XHRs
        // See b/131755324 for more info.
        if (e.key === 'Escape') {
          e.stopPropagation();
          e.preventDefault();
        }
      });
    </script><meta name="referrer" content="origin"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Welcome To Colab - Colab</title><link href="./untitled10_files/css2" rel="stylesheet"><link href="./untitled10_files/css" rel="stylesheet"><link rel="search" type="application/opensearchdescription+xml" href="https://colab.research.google.com/opensearch.xml" title="Google Colaboratory"><style>.gb_ib:not(.gb_kd){font:13px/27px Roboto,Arial,sans-serif;z-index:986}@-webkit-keyframes gb__a{0%{opacity:0}50%{opacity:1}}@keyframes gb__a{0%{opacity:0}50%{opacity:1}}a.gb_za{border:none;color:#4285f4;cursor:default;font-weight:bold;outline:none;position:relative;text-align:center;text-decoration:none;text-transform:uppercase;white-space:nowrap;-webkit-user-select:none}a.gb_za:hover:after,a.gb_za:focus:after{background-color:rgba(0,0,0,.12);content:"";height:100%;left:0;position:absolute;top:0;width:100%}a.gb_za:hover,a.gb_za:focus{text-decoration:none}a.gb_za:active{background-color:rgba(153,153,153,.4);text-decoration:none}a.gb_Aa{background-color:#4285f4;color:#fff}a.gb_Aa:active{background-color:#0043b2}.gb_Ba{box-shadow:0 1px 1px rgba(0,0,0,.16)}.gb_za,.gb_Aa,.gb_Ca,.gb_Da{display:inline-block;line-height:28px;padding:0 12px;border-radius:2px}.gb_Ca{background:#f8f8f8;border:1px solid #c6c6c6}.gb_Da{background:#f8f8f8}.gb_Ca,#gb a.gb_Ca.gb_Ca,.gb_Da{color:#666;cursor:default;text-decoration:none}#gb a.gb_Da{cursor:default;text-decoration:none}.gb_Da{border:1px solid #4285f4;font-weight:bold;outline:none;background:#4285f4;background:-webkit-gradient(linear,left top,left bottom,from(top),color-stop(#4387fd),to(#4683ea));background:-webkit-linear-gradient(top,#4387fd,#4683ea);background:linear-gradient(top,#4387fd,#4683ea);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr=#4387fd,endColorstr=#4683ea,GradientType=0)}#gb a.gb_Da{color:#fff}.gb_Da:hover{box-shadow:0 1px 0 rgba(0,0,0,.15)}.gb_Da:active{box-shadow:inset 0 2px 0 rgba(0,0,0,.15);background:#3c78dc;background:-webkit-gradient(linear,left top,left bottom,from(top),color-stop(#3c7ae4),to(#3f76d3));background:-webkit-linear-gradient(top,#3c7ae4,#3f76d3);background:linear-gradient(top,#3c7ae4,#3f76d3);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr=#3c7ae4,endColorstr=#3f76d3,GradientType=0)}#gb .gb_Ea{background:#fff;border:1px solid #dadce0;color:#1a73e8;display:inline-block;text-decoration:none}#gb .gb_Ea:hover{background:#f8fbff;border-color:#dadce0;color:#174ea6}#gb .gb_Ea:focus{background:#f4f8ff;color:#174ea6;outline:1px solid #174ea6}#gb .gb_Ea:active,#gb .gb_Ea:focus:active{background:#ecf3fe;color:#174ea6}#gb .gb_Ea.gb_i{background:transparent;border:1px solid #5f6368;color:#8ab4f8;text-decoration:none}#gb .gb_Ea.gb_i:hover{background:rgba(255,255,255,.04);color:#e8eaed}#gb .gb_Ea.gb_i:focus{background:rgba(232,234,237,.12);color:#e8eaed;outline:1px solid #e8eaed}#gb .gb_Ea.gb_i:active,#gb .gb_Ea.gb_i:focus:active{background:rgba(232,234,237,.1);color:#e8eaed}.gb_r{display:none!important}.gb_3a{visibility:hidden}.gb_y{display:inline-block;vertical-align:middle}.gb_Sd .gb_q{bottom:-3px;right:-5px}.gb_f{position:relative}.gb_d{display:inline-block;outline:none;vertical-align:middle;border-radius:2px;box-sizing:border-box;height:40px;width:40px;cursor:pointer;text-decoration:none}#gb#gb a.gb_d{cursor:pointer;text-decoration:none}.gb_d,a.gb_d{color:#000}.gb_if{border-color:transparent;border-bottom-color:#fff;border-style:dashed dashed solid;border-width:0 8.5px 8.5px;display:none;position:absolute;left:11.5px;top:33px;z-index:1;height:0;width:0;-webkit-animation:gb__a .2s;animation:gb__a .2s}.gb_jf{border-color:transparent;border-style:dashed dashed solid;border-width:0 8.5px 8.5px;display:none;position:absolute;left:11.5px;z-index:1;height:0;width:0;-webkit-animation:gb__a .2s;animation:gb__a .2s;border-bottom-color:rgba(0,0,0,.2);top:32px}x:-o-prefocus,div.gb_jf{border-bottom-color:#ccc}.gb_7{background:#fff;border:1px solid #ccc;border-color:rgba(0,0,0,.2);color:#000;-webkit-box-shadow:0 2px 10px rgba(0,0,0,.2);box-shadow:0 2px 10px rgba(0,0,0,.2);display:none;outline:none;overflow:hidden;position:absolute;right:8px;top:62px;-webkit-animation:gb__a .2s;animation:gb__a .2s;border-radius:2px;-webkit-user-select:text}.gb_y.gb_Na .gb_if,.gb_y.gb_Na .gb_jf,.gb_y.gb_Na .gb_7,.gb_Na.gb_7{display:block}.gb_y.gb_Na.gb_kf .gb_if,.gb_y.gb_Na.gb_kf .gb_jf{display:none}.gb_Td{position:absolute;right:8px;top:62px;z-index:-1}.gb_9a .gb_if,.gb_9a .gb_jf,.gb_9a .gb_7{margin-top:-10px}.gb_y:first-child,#gbsfw:first-child+.gb_y{padding-left:4px}.gb_Ra.gb_Ud .gb_y:first-child{padding-left:0}.gb_Vd{position:relative}.gb_v.gb_Cd.gb_gb.gb_rd{margin:0 12px;padding:0}.gb_v .gb_d{position:relative}.gb_v .gb_y{margin:0 4px;padding:4px}.gb_v .gb_Wd{display:inline-block}.gb_v a.gb_nd{-webkit-box-align:center;-webkit-align-items:center;-webkit-align-items:center;align-items:center;-webkit-border-radius:100px;border-radius:100px;border:0;background:#0b57d0;color:#fff;display:-webkit-inline-box;display:-webkit-inline-flex;display:-webkit-inline-box;display:-webkit-inline-flex;display:inline-flex;font-size:14px;font-weight:500;height:40px;white-space:nowrap;width:auto}.gb_v a.gb_d.gb_nd{margin:0 4px;padding:4px 24px 4px 24px}.gb_v a.gb_nd.gb_Xd{padding:9px 12px 9px 16px}.gb_v a.gb_nd.gb_Zd{background:transparent;border:1px solid #747775;color:#0b57d0;outline:0}.gb_v .gb_u{fill:#0b57d0}.gb_v .gb_0d{fill:#0b57d0;margin-left:8px}.gb_v .gb_0d circle{fill:#fff}.gb_v .gb_nd .gb_Ld{-webkit-box-flex:1;-webkit-flex-grow:1;-webkit-box-flex:1;box-flex:1;-webkit-flex-grow:1;flex-grow:1;text-align:center}.gb_v .gb_nd:hover{background:#3763cd}.gb_v .gb_nd:hover .gb_0d{fill:#3763cd}.gb_v .gb_nd:focus,.gb_v .gb_nd:active,.gb_v .gb_nd:focus:hover,.gb_v .gb_nd[aria-expanded=true],.gb_v .gb_nd:hover[aria-expanded=true]{background:#416acf}.gb_v .gb_nd:focus .gb_0d,.gb_v .gb_nd:active .gb_0d,.gb_v .gb_nd:focus:hover .gb_0d,.gb_v .gb_nd[aria-expanded=true] .gb_0d,.gb_v .gb_nd:hover[aria-expanded=true] .gb_0d{fill:#416acf}.gb_v .gb_nd:focus,.gb_v .gb_nd:active,.gb_v .gb_nd[aria-expanded=true]{-webkit-box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3);box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3)}.gb_v .gb_nd:focus-visible{outline:1px solid #416acf;outline-offset:2px}.gb_v .gb_Ia:focus-visible{outline:1px solid #416acf}.gb_v .gb_i.gb_nd{background:#a8c7fa;color:#062e6f}.gb_v .gb_i.gb_nd .gb_0d{fill:#a8c7fa}.gb_v .gb_i.gb_nd .gb_0d circle{fill:#062e6f}.gb_v .gb_i.gb_nd:hover{background:#b4cbf6}.gb_v .gb_i.gb_nd:hover .gb_0d{fill:#b4cbf6}.gb_v .gb_i.gb_nd:focus,.gb_v .gb_i.gb_nd:focus:hover,.gb_v .gb_i.gb_nd:active,.gb_v .gb_i.gb_nd[aria-expanded=true],.gb_v .gb_i.gb_nd:hover[aria-expanded=true]{background:#b8cdf7}.gb_v .gb_i.gb_nd:focus .gb_0d,.gb_v .gb_i.gb_nd:focus:hover .gb_0d,.gb_v .gb_i.gb_nd:active .gb_0d,.gb_v .gb_i.gb_nd[aria-expanded=true] .gb_0d,.gb_v .gb_i.gb_nd:hover[aria-expanded=true] .gb_0d{fill:#b8cdf7}.gb_v .gb_i.gb_nd:focus-visible{outline-color:#b8cdf7}.gb_v .gb_i.gb_nd:focus,.gb_v .gb_i.gb_nd:active,.gb_v .gb_i.gb_nd[aria-expanded=true]{-webkit-box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3);box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3)}.gb_v .gb_nd.gb_Zd:hover,.gb_v .gb_nd.gb_Zd:focus,.gb_v .gb_nd.gb_Zd[aria-expanded=true],.gb_v .gb_nd.gb_Zd:hover[aria-expanded=true]{background:rgba(11,87,208,.08);-webkit-box-shadow:none;box-shadow:none}.gb_v .gb_nd.gb_Zd:active{background:rgba(11,87,208,.12);-webkit-box-shadow:none;box-shadow:none}.gb_v .gb_nd.gb_Zd:focus-visible{border-color:#0b57d0;outline:0}.gb_v .gb_i.gb_nd.gb_Zd{background:transparent;color:#a8c7fa}.gb_v .gb_i.gb_nd.gb_Zd:hover,.gb_v .gb_i.gb_nd.gb_Zd:focus,.gb_v .gb_i.gb_nd.gb_Zd[aria-expanded=true],.gb_v .gb_i.gb_nd.gb_Zd:hover[aria-expanded=true]{background:rgba(168,199,250,.08);-webkit-box-shadow:none;box-shadow:none}.gb_v .gb_i.gb_nd.gb_Zd:active{background:rgba(168,199,250,.12);-webkit-box-shadow:none;box-shadow:none}.gb_v .gb_i.gb_nd.gb_Zd:focus-visible{border-color:#a8c7fa;outline:0}.gb_i .gb_v .gb_u{fill:#a8c7fa}.gb_i .gb_v .gb_Ia:focus-visible{outline-color:#a8c7fa}.gb_7c .gb_Vd,.gb_md .gb_Vd{float:right}.gb_d{padding:8px;cursor:pointer}.gb_d:after{content:"";position:absolute;top:-4px;bottom:-4px;left:-4px;right:-4px}.gb_Ra .gb_ne:not(.gb_za):focus img{background-color:rgba(0,0,0,.2);outline:none;-webkit-border-radius:50%;border-radius:50%}.gb_1d button svg,.gb_d{-webkit-border-radius:50%;border-radius:50%}.gb_1d button:focus:not(:focus-visible) svg,.gb_1d button:hover svg,.gb_1d button:active svg,.gb_d:focus:not(:focus-visible),.gb_d:hover,.gb_d:active,.gb_d[aria-expanded=true]{outline:none}.gb_Qc .gb_1d.gb_we button:focus-visible svg,.gb_1d button:focus-visible svg,.gb_d:focus-visible{outline:1px solid #202124}.gb_Qc .gb_1d button:focus-visible svg,.gb_Qc .gb_d:focus-visible{outline:1px solid #f1f3f4}@media (forced-colors:active){.gb_Qc .gb_1d.gb_we button:focus-visible svg,.gb_1d button:focus-visible svg,.gb_Qc .gb_1d button:focus-visible svg{outline:1px solid currentcolor}}.gb_Qc .gb_1d.gb_we button:focus svg,.gb_Qc .gb_1d.gb_we button:focus:hover svg,.gb_1d button:focus svg,.gb_1d button:focus:hover svg,.gb_d:focus,.gb_d:focus:hover{background-color:rgba(60,64,67,.1)}.gb_Qc .gb_1d.gb_we button:active svg,.gb_1d button:active svg,.gb_d:active{background-color:rgba(60,64,67,.12)}.gb_Qc .gb_1d.gb_we button:hover svg,.gb_1d button:hover svg,.gb_d:hover{background-color:rgba(60,64,67,.08)}.gb_Fa .gb_d.gb_Ia:hover{background-color:transparent}.gb_d[aria-expanded=true],.gb_d:hover[aria-expanded=true]{background-color:rgba(95,99,104,.24)}.gb_d[aria-expanded=true] .gb_h{fill:#5f6368;opacity:1}.gb_Qc .gb_1d button:hover svg,.gb_Qc .gb_d:hover{background-color:rgba(232,234,237,.08)}.gb_Qc .gb_1d button:focus svg,.gb_Qc .gb_1d button:focus:hover svg,.gb_Qc .gb_d:focus,.gb_Qc .gb_d:focus:hover{background-color:rgba(232,234,237,.1)}.gb_Qc .gb_1d button:active svg,.gb_Qc .gb_d:active{background-color:rgba(232,234,237,.12)}.gb_Qc .gb_d[aria-expanded=true],.gb_Qc .gb_d:hover[aria-expanded=true]{background-color:rgba(255,255,255,.12)}.gb_Qc .gb_d[aria-expanded=true] .gb_h{fill:#fff;opacity:1}.gb_y{padding:4px}.gb_Ra.gb_Ud .gb_y{padding:4px 2px}.gb_Ra.gb_Ud .gb_b.gb_y{padding-left:6px}.gb_7{z-index:991;line-height:normal}.gb_7.gb_2d{left:0;right:auto}@media (max-width:350px){.gb_7.gb_2d{left:0}}.gb_3d .gb_7{top:56px}.gb_k .gb_d,.gb_6 .gb_k .gb_d{background-position:-64px -29px}.gb_M .gb_k .gb_d{background-position:-29px -29px;opacity:1}.gb_k .gb_d,.gb_k .gb_d:hover,.gb_k .gb_d:focus{opacity:1}.gb_n{display:none}@media screen and (max-width:319px){.gb_sd:not(.gb_xd) .gb_k{display:none;visibility:hidden}}.gb_q{display:none}.gb_fd{font-family:Google Sans,Roboto,Helvetica,Arial,sans-serif;font-size:20px;font-weight:400;letter-spacing:0.25px;line-height:48px;margin-bottom:2px;opacity:1;overflow:hidden;padding-left:16px;position:relative;text-overflow:ellipsis;vertical-align:middle;top:2px;white-space:nowrap;-webkit-flex:1 1 auto;-webkit-box-flex:1;flex:1 1 auto}.gb_fd.gb_gd{color:#3c4043}.gb_Ra.gb_Sa .gb_fd{margin-bottom:0}.gb_hd.gb_id .gb_fd{padding-left:4px}.gb_Ra.gb_Sa .gb_jd{position:relative;top:-2px}.gb_Ra{color:black;min-width:160px;position:relative;-webkit-transition:box-shadow 250ms;transition:box-shadow 250ms}.gb_Ra.gb_Zc{min-width:120px}.gb_Ra.gb_qd .gb_rd{display:none}.gb_Ra.gb_qd .gb_sd{height:56px}header.gb_Ra{display:block}.gb_Ra svg{fill:currentColor}.gb_td{position:fixed;top:0;width:100%}.gb_ud{-webkit-box-shadow:0 4px 5px 0 rgba(0,0,0,.14),0 1px 10px 0 rgba(0,0,0,.12),0 2px 4px -1px rgba(0,0,0,.2);box-shadow:0 4px 5px 0 rgba(0,0,0,.14),0 1px 10px 0 rgba(0,0,0,.12),0 2px 4px -1px rgba(0,0,0,.2)}.gb_vd{height:64px}.gb_sd{-webkit-box-sizing:border-box;box-sizing:border-box;position:relative;width:100%;display:-webkit-box;display:-webkit-flex;display:flex;-webkit-box-pack:space-between;-webkit-justify-content:space-between;justify-content:space-between;min-width:-webkit-min-content;min-width:min-content}.gb_Ra:not(.gb_Sa) .gb_sd{padding:8px}.gb_Ra.gb_wd .gb_sd{-webkit-flex:1 0 auto;-webkit-box-flex:1;flex:1 0 auto}.gb_Ra .gb_sd.gb_xd.gb_yd{min-width:0}.gb_Ra.gb_Sa .gb_sd{padding:4px;padding-left:8px;min-width:0}.gb_rd{height:48px;vertical-align:middle;white-space:nowrap;-webkit-box-align:center;-webkit-align-items:center;align-items:center;display:-webkit-box;display:-webkit-flex;display:flex;-webkit-user-select:none}.gb_Ad>.gb_rd{display:table-cell;width:100%}.gb_hd{padding-right:30px;box-sizing:border-box;-webkit-flex:1 0 auto;-webkit-box-flex:1;flex:1 0 auto}.gb_Ra.gb_Sa .gb_hd{padding-right:14px}.gb_Bd{-webkit-flex:1 1 100%;-webkit-box-flex:1;flex:1 1 100%}.gb_Bd>:only-child{display:inline-block}.gb_Cd.gb_8c{padding-left:4px}.gb_Cd.gb_Dd,.gb_Ra.gb_wd .gb_Cd,.gb_Ra.gb_Sa:not(.gb_md) .gb_Cd{padding-left:0}.gb_Ra.gb_Sa .gb_Cd.gb_Dd{padding-right:0}.gb_Ra.gb_Sa .gb_Cd.gb_Dd .gb_Fa{margin-left:10px}.gb_8c{display:inline}.gb_Ra.gb_2c .gb_Cd.gb_Ed,.gb_Ra.gb_md .gb_Cd.gb_Ed{padding-left:2px}.gb_fd{display:inline-block}.gb_Cd{-webkit-box-sizing:border-box;box-sizing:border-box;height:48px;line-height:normal;padding:0 4px;padding-left:30px;-webkit-flex:0 0 auto;-webkit-box-flex:0;flex:0 0 auto;-webkit-box-pack:flex-end;-webkit-justify-content:flex-end;justify-content:flex-end}.gb_md{height:48px}.gb_Ra.gb_md{min-width:auto}.gb_md .gb_Cd{float:right;padding-left:32px}.gb_md .gb_Cd.gb_Fd{padding-left:0}.gb_Hd{font-size:14px;max-width:200px;overflow:hidden;padding:0 12px;text-overflow:ellipsis;white-space:nowrap;-webkit-user-select:text}.gb_ld{-webkit-transition:background-color .4s;-webkit-transition:background-color .4s;transition:background-color .4s}.gb_Nd{color:black}.gb_Qc{color:white}.gb_Ra a,.gb_Vc a{color:inherit}.gb_W{color:rgba(0,0,0,.87)}.gb_Ra svg,.gb_Vc svg,.gb_hd .gb_pd,.gb_7c .gb_pd{color:#5f6368;opacity:1}.gb_Qc svg,.gb_Vc.gb_0c svg,.gb_Qc .gb_hd .gb_pd,.gb_Qc .gb_hd .gb_Pc,.gb_Qc .gb_hd .gb_jd,.gb_Vc.gb_0c .gb_pd{color:rgba(255,255,255,.87)}.gb_Qc .gb_hd .gb_Oc:not(.gb_Od){opacity:.87}.gb_gd{color:inherit;opacity:1;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased}.gb_Qc .gb_gd,.gb_Nd .gb_gd{opacity:1}.gb_Id{position:relative}.gb_Jd{font-family:arial,sans-serif;line-height:normal;padding-right:15px}a.gb_J,span.gb_J{color:rgba(0,0,0,.87);text-decoration:none}.gb_Qc a.gb_J,.gb_Qc span.gb_J{color:white}a.gb_J:focus{outline-offset:2px}a.gb_J:hover{text-decoration:underline}.gb_K{display:inline-block;padding-left:15px}.gb_K .gb_J{display:inline-block;line-height:24px;vertical-align:middle}.gb_Pd{font-family:Google Sans,Roboto,Helvetica,Arial,sans-serif;font-weight:500;font-size:14px;letter-spacing:.25px;line-height:16px;margin-left:10px;margin-right:8px;min-width:96px;padding:9px 23px;text-align:center;vertical-align:middle;border-radius:4px;box-sizing:border-box}.gb_Ra.gb_md .gb_Pd{margin-left:8px}#gb a.gb_Da.gb_Pd{cursor:pointer}.gb_Da.gb_Pd:hover{background:#1b66c9;-webkit-box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3);box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3)}.gb_Da.gb_Pd:focus,.gb_Da.gb_Pd:hover:focus{background:#1c5fba;-webkit-box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3);box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3)}.gb_Da.gb_Pd:active{background:#1b63c1;-webkit-box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3);box-shadow:0 1px 3px 1px rgba(66,64,67,.15),0 1px 2px 0 rgba(60,64,67,.3)}.gb_Pd{background:#1a73e8;border:1px solid transparent}.gb_Ra.gb_Sa .gb_Pd{padding:9px 15px;min-width:80px}.gb_Kd{text-align:left}#gb .gb_Qc a.gb_Pd:not(.gb_i),#gb.gb_Qc a.gb_Pd:not(.gb_Qd){background:#fff;border-color:#dadce0;-webkit-box-shadow:none;box-shadow:none;color:#1a73e8}#gb a.gb_Da.gb_i.gb_Pd{background:#8ab4f8;border:1px solid transparent;-webkit-box-shadow:none;box-shadow:none;color:#202124}#gb .gb_Qc a.gb_Pd:hover:not(.gb_i),#gb.gb_Qc a.gb_Pd:not(.gb_Qd):hover{background:#f8fbff;border-color:#cce0fc}#gb a.gb_Da.gb_i.gb_Pd:hover{background:#93baf9;border-color:transparent;-webkit-box-shadow:0 1px 3px 1px rgba(0,0,0,.15),0 1px 2px rgba(0,0,0,.3);box-shadow:0 1px 3px 1px rgba(0,0,0,.15),0 1px 2px rgba(0,0,0,.3)}#gb .gb_Qc a.gb_Pd:focus:not(.gb_i),#gb .gb_Qc a.gb_Pd:focus:hover:not(.gb_i),#gb.gb_Qc a.gb_Pd:focus:not(.gb_i),#gb.gb_Qc a.gb_Pd:focus:hover:not(.gb_i){background:#f4f8ff;outline:1px solid #c9ddfc}#gb a.gb_Da.gb_i.gb_Pd:focus,#gb a.gb_Da.gb_i.gb_Pd:focus:hover{background:#a6c6fa;border-color:transparent;-webkit-box-shadow:none;box-shadow:none}#gb .gb_Qc a.gb_Pd:active:not(.gb_i),#gb.gb_Qc a.gb_Pd:not(.gb_Qd):active{background:#ecf3fe}#gb a.gb_Da.gb_i.gb_Pd:active{background:#a1c3f9;-webkit-box-shadow:0 1px 2px rgba(60,64,67,.3),0 2px 6px 2px rgba(60,64,67,.15);box-shadow:0 1px 2px rgba(60,64,67,.3),0 2px 6px 2px rgba(60,64,67,.15)}.gb_l{display:none}@media screen and (max-width:319px){.gb_sd:not(.gb_xd) .gb_k{display:none;visibility:hidden}}.gb_Fa{background-color:rgba(255,255,255,.88);border:1px solid #dadce0;-webkit-box-sizing:border-box;box-sizing:border-box;cursor:pointer;display:inline-block;max-height:48px;overflow:hidden;outline:none;padding:0;vertical-align:middle;width:134px;-webkit-border-radius:8px;border-radius:8px}.gb_Fa.gb_i{background-color:transparent;border:1px solid #5f6368}.gb_Ma{display:inherit}.gb_Fa.gb_i .gb_Ma{background:#fff;-webkit-border-radius:4px;border-radius:4px;display:inline-block;left:8px;margin-right:5px;position:relative;padding:3px;top:-1px}.gb_Fa:hover{border:1px solid #d2e3fc;background-color:rgba(248,250,255,.88)}.gb_Fa.gb_i:hover{background-color:rgba(241,243,244,.04);border:1px solid #5f6368}.gb_Fa:focus-visible,.gb_Fa:focus{background-color:#fff;outline:1px solid #202124;-webkit-box-shadow:0 1px 2px 0 rgba(60,64,67,.3),0 1px 3px 1px rgba(60,64,67,.15);box-shadow:0 1px 2px 0 rgba(60,64,67,.3),0 1px 3px 1px rgba(60,64,67,.15)}.gb_Fa.gb_i:focus-visible,.gb_Fa.gb_i:focus{background-color:rgba(241,243,244,.12);outline:1px solid #f1f3f4;-webkit-box-shadow:0 1px 3px 1px rgba(0,0,0,.15),0 1px 2px 0 rgba(0,0,0,.3);box-shadow:0 1px 3px 1px rgba(0,0,0,.15),0 1px 2px 0 rgba(0,0,0,.3)}.gb_Fa.gb_i:active,.gb_Fa.gb_Na.gb_i:focus{background-color:rgba(241,243,244,.1);border:1px solid #5f6368}.gb_Oa{display:inline-block;padding-bottom:2px;padding-left:7px;padding-top:2px;text-align:center;vertical-align:middle;line-height:32px;width:78px}.gb_Fa.gb_i .gb_Oa{line-height:26px;margin-left:0;padding-bottom:0;padding-left:0;padding-top:0;width:72px}.gb_Oa.gb_Pa{background-color:#f1f3f4;-webkit-border-radius:4px;border-radius:4px;margin-left:8px;padding-left:0;line-height:30px}.gb_Oa.gb_Pa .gb_Qa{vertical-align:middle}.gb_Ra:not(.gb_Sa) .gb_Fa{margin-left:10px;margin-right:4px}.gb_Ta{max-height:32px;width:78px}.gb_Fa.gb_i .gb_Ta{max-height:26px;width:72px}.gb_p{-webkit-background-size:32px 32px;background-size:32px 32px;border:0;-webkit-border-radius:50%;border-radius:50%;display:block;margin:0px;position:relative;height:32px;width:32px;z-index:0}.gb_4a{background-color:#e8f0fe;border:1px solid rgba(32,33,36,.08);position:relative}.gb_4a.gb_p{height:30px;width:30px}.gb_4a.gb_p:hover,.gb_4a.gb_p:active{-webkit-box-shadow:none;box-shadow:none}.gb_5a{background:#fff;border:none;-webkit-border-radius:50%;border-radius:50%;bottom:2px;-webkit-box-shadow:0px 1px 2px 0px rgba(60,64,67,.30),0px 1px 3px 1px rgba(60,64,67,.15);box-shadow:0px 1px 2px 0px rgba(60,64,67,.30),0px 1px 3px 1px rgba(60,64,67,.15);height:14px;margin:2px;position:absolute;right:0;width:14px}.gb_6a{color:#1f71e7;font:400 22px/32px Google Sans,Roboto,Helvetica,Arial,sans-serif;text-align:center;text-transform:uppercase}@media (-webkit-min-device-pixel-ratio:1.25),(min-resolution:1.25dppx),(min-device-pixel-ratio:1.25){.gb_p::before,.gb_7a::before{display:inline-block;-webkit-transform:scale(0.5);-webkit-transform:scale(0.5);transform:scale(0.5);-webkit-transform-origin:left 0;-webkit-transform-origin:left 0;transform-origin:left 0}.gb_O .gb_7a::before{-webkit-transform:scale(scale(0.416666667));-webkit-transform:scale(scale(0.416666667));transform:scale(scale(0.416666667))}}.gb_p:hover,.gb_p:focus{-webkit-box-shadow:0 1px 0 rgba(0,0,0,.15);box-shadow:0 1px 0 rgba(0,0,0,.15)}.gb_p:active{-webkit-box-shadow:inset 0 2px 0 rgba(0,0,0,.15);box-shadow:inset 0 2px 0 rgba(0,0,0,.15)}.gb_p:active::after{background:rgba(0,0,0,.1);-webkit-border-radius:50%;border-radius:50%;content:"";display:block;height:100%}.gb_8a{cursor:pointer;line-height:40px;min-width:30px;opacity:.75;overflow:hidden;vertical-align:middle;text-overflow:ellipsis}.gb_d.gb_8a{width:auto}.gb_8a:hover,.gb_8a:focus{opacity:.85}.gb_9a .gb_8a,.gb_9a .gb_ab{line-height:26px}#gb#gb.gb_9a a.gb_8a,.gb_9a .gb_ab{font-size:11px;height:auto}.gb_bb{border-top:4px solid #000;border-left:4px dashed transparent;border-right:4px dashed transparent;display:inline-block;margin-left:6px;opacity:.75;vertical-align:middle}.gb_Ia:hover .gb_bb{opacity:.85}.gb_Fa>.gb_b{padding:3px 3px 3px 4px}.gb_cb.gb_3a{color:#fff}.gb_M .gb_8a,.gb_M .gb_bb{opacity:1}#gb#gb.gb_M.gb_M a.gb_8a,#gb#gb .gb_M.gb_M a.gb_8a{color:#fff}.gb_M.gb_M .gb_bb{border-top-color:#fff;opacity:1}.gb_6 .gb_p:hover,.gb_M .gb_p:hover,.gb_6 .gb_p:focus,.gb_M .gb_p:focus{-webkit-box-shadow:0 1px 0 rgba(0,0,0,.15),0 1px 2px rgba(0,0,0,.2);box-shadow:0 1px 0 rgba(0,0,0,.15),0 1px 2px rgba(0,0,0,.2)}.gb_db .gb_b,.gb_eb .gb_b{position:absolute;right:1px}.gb_b.gb_L,.gb_fb.gb_L,.gb_Ia.gb_L{-webkit-flex:0 1 auto;-webkit-box-flex:0;flex:0 1 auto}.gb_gb.gb_hb .gb_8a{width:30px!important}.gb_o{height:40px;position:absolute;right:-5px;top:-5px;width:40px}.gb_ib .gb_o,.gb_jb .gb_o{right:0;top:0}.gb_b .gb_d{padding:4px}.gb_s{display:none}sentinel{}</style><script nonce="">;this.gbar_={CONFIG:[[[0,"www.gstatic.com","og.qtm.en_US.oT1FwJRCVC4.2019.O","com","en","425",0,[4,2,"","","","624795372","0"],null,"XtYpZri2M-rSp84P2JSk0AE",null,0,"og.qtm.T5bVtXo12IQ.L.W.O","AA2YrTvBynad-nWEy1xIb9j1w6LpLOF6IQ","AA2YrTssrVR1lBtzoy_MObv1DSp-vWG36A","",2,1,200,"USA",null,null,"425","425",1,null,null,114591953,0],null,[1,0.1000000014901161,2,1],null,[1,0,0,null,"0","luckychamp141@gmail.com","","ABBHTIsTEy15hu9eMsFs4CuT5Qnkrj7XGtLU0745Zkti-jAFKNDkAOXI_WKurUfc7UezK5jG0Z4_vEAbCp2oBGpyAnsBlGRj1A",0,0,0],[0,0,"",1,0,0,0,0,0,0,null,0,0,null,0,0,null,null,0,0,0,"","","","","","",null,0,0,0,0,0,null,null,null,"rgba(32,33,36,1)","rgba(255,255,255,1)",0,0,0,null,null,1,0,0],["%1$s (default)","Brand account",1,"%1$s (delegated)",1,null,83,"https://colab.research.google.com/?authuser=$authuser",null,null,null,1,"https://accounts.google.com/ListAccounts?listPages=0\u0026pid=425\u0026gpsia=1\u0026source=ogb\u0026atic=1\u0026mo=1\u0026mn=1\u0026hl=en\u0026ts=250",0,"dashboard",null,null,null,null,"Profile","",1,null,"Signed out","https://accounts.google.com/AccountChooser?source=ogb\u0026continue=$continue\u0026Email=$email\u0026ec=GAhAqQM","https://accounts.google.com/RemoveLocalAccount?source=ogb","Remove","Sign in",0,1,1,0,1,1,0,null,null,null,"Session expired",null,null,null,"Visitor",null,"Default","Delegated","Sign out of all accounts",0,null,null,0,null,null,"myaccount.google.com","https",0,1,0],null,["1","gci_91f30755d6a6b787dcc2a4062e6e9824.js","googleapis.client:gapi.iframes","0","en"],null,null,null,null,["m;/_/scs/abc-static/_/js/k=gapi.gapi.en.SCWmpDDGjPk.O/am=AAAC/d=1/rs=AHpOoo_Pl64J0IIHlj2zBtEJ3ZwdaJC3HA/m=__features__","https://apis.google.com","","","1","",null,1,"es_plusone_gc_20240331.0_p2","en",null,0],[0.009999999776482582,"com","425",[null,"","0",null,1,5184000,null,null,"",null,null,null,null,null,0,null,0,null,1,0,0,0,null,null,0,0,null,0,0,0,0,0],null,null,null,0,null,null,["5061451","google\\.(com|ru|ca|by|kz|com\\.mx|com\\.tr)$",1]],[1,1,null,40400,425,"USA","en","624795372.0",8,0.009999999776482582,1,0,null,null,null,null,"3700949,3701310",null,null,null,"XtYpZri2M-rSp84P2JSk0AE",0,0,0,null,2,5,"nn",7,0,0,1,1,1,114591953,0],[[null,null,null,"https://www.gstatic.com/og/_/js/k=og.qtm.en_US.oT1FwJRCVC4.2019.O/rt=j/m=qabr,qgl,q_dnp,qcwid,qbd,qapid,qrcd,q_dg/exm=qaaw,qadd,qaid,qein,qhaw,qhba,qhbr,qhch,qhga,qhid,qhin/d=1/ed=1/rs=AA2YrTvBynad-nWEy1xIb9j1w6LpLOF6IQ"],[null,null,null,"https://www.gstatic.com/og/_/ss/k=og.qtm.T5bVtXo12IQ.L.W.O/m=qcwid/excm=qaaw,qadd,qaid,qein,qhaw,qhba,qhbr,qhch,qhga,qhid,qhin/d=1/ed=1/ct=zgms/rs=AA2YrTssrVR1lBtzoy_MObv1DSp-vWG36A"]],null,null,null,[[[null,null,[null,null,null,"https://ogs.google.com/u/0/widget/account?amb=1\u0026sea=1"],0,414,436,57,4,1,0,0,65,66,8000,"https://accounts.google.com/SignOutOptions?hl=en\u0026continue=https://colab.research.google.com/\u0026ec=GBRAqQM",68,2,null,null,1,113,"Something went wrong.%1$s Refresh to try again or %2$schoose another account%3$s.",3,null,null,75,0,null,null,null,null,null,null,null,"/widget/account",["https","myaccount.google.com",0,32,83,0],0,0,1,["Critical security alert","Important account alert"],0,1,null,1,1,0,0,0,0,0]],null,null,"425","425",1,0,null,"en",0,["https://colab.research.google.com/?authuser=$authuser","https://accounts.google.com/AddSession?hl=en\u0026continue=https://colab.research.google.com/\u0026ec=GAlAqQM","https://accounts.google.com/Logout?hl=en\u0026continue=https://colab.research.google.com/\u0026timeStmp=1714017886\u0026secTok=.AG5fkS8rhxll8FcDBBeu-SQo9Vv6zNbz_g\u0026ec=GAdAqQM","https://accounts.google.com/ListAccounts?listPages=0\u0026pid=425\u0026gpsia=1\u0026source=ogb\u0026atic=1\u0026mo=1\u0026mn=1\u0026hl=en\u0026ts=250",0,0,"",0,0,null,0,0,"https://accounts.google.com/ServiceLogin?passive=true\u0026continue=https%3A%2F%2Fcolab.research.google.com%2F\u0026ec=GAZAqQM",0,0],0,0,0,null,0,null,0,0],null,[["mousedown","touchstart","touchmove","wheel","keydown"],300000],[[null,null,null,"https://accounts.google.com/RotateCookiesPage"],3,null,null,null,0,0]]],};this.gbar_=this.gbar_||{};(function(_){var window=this;
try{
_._F_toggles_initialize=function(a){("undefined"!==typeof globalThis?globalThis:"undefined"!==typeof self?self:this)._F_toggles=a||[]};(0,_._F_toggles_initialize)([]);
/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var fa,la,na,oa,xa,ya,za,Ba,Ca,Da,Ha,Xa,Wa,Za,ab,$a,bb,cb,fb,lb,nb,ob,pb,qb;_.aa=function(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,_.aa);else{const c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));void 0!==b&&(this.cause=b)};_.ba=function(a){_.q.setTimeout(()=>{throw a;},0)};_.ca=function(){var a=_.q.navigator;return a&&(a=a.userAgent)?a:""};fa=function(a){return da?ea?ea.brands.some(({brand:b})=>b&&-1!=b.indexOf(a)):!1:!1};_.u=function(a){return-1!=_.ca().indexOf(a)};
_.ha=function(){return da?!!ea&&0<ea.brands.length:!1};_.ia=function(){return _.ha()?!1:_.u("Opera")};_.ja=function(){return _.ha()?!1:_.u("Trident")||_.u("MSIE")};_.ka=function(){return _.u("Firefox")||_.u("FxiOS")};_.ma=function(){return _.u("Safari")&&!(la()||(_.ha()?0:_.u("Coast"))||_.ia()||(_.ha()?0:_.u("Edge"))||(_.ha()?fa("Microsoft Edge"):_.u("Edg/"))||(_.ha()?fa("Opera"):_.u("OPR"))||_.ka()||_.u("Silk")||_.u("Android"))};
la=function(){return _.ha()?fa("Chromium"):(_.u("Chrome")||_.u("CriOS"))&&!(_.ha()?0:_.u("Edge"))||_.u("Silk")};na=function(){return da?!!ea&&!!ea.platform:!1};oa=function(){return _.u("iPhone")&&!_.u("iPod")&&!_.u("iPad")};_.pa=function(){return oa()||_.u("iPad")||_.u("iPod")};_.qa=function(){return na()?"macOS"===ea.platform:_.u("Macintosh")};_.sa=function(a,b){return 0<=_.ra(a,b)};
_.ta=function(a){let b="",c=0;const d=a.length-10240;for(;c<d;)b+=String.fromCharCode.apply(null,a.subarray(c,c+=10240));b+=String.fromCharCode.apply(null,c?a.subarray(c):a);return btoa(b)};_.ua=function(a){return null!=a&&a instanceof Uint8Array};_.va=function(a){return Array.prototype.slice.call(a)};xa=function(a,b){_.wa(b,(a|0)&-14591)};ya=function(a,b){_.wa(b,(a|34)&-14557)};za=function(a){a=a>>14&1023;return 0===a?536870912:a};Ba=function(a){return!(!a||"object"!==typeof a||a.i!==Aa)};
Ca=function(a){return null!==a&&"object"===typeof a&&!Array.isArray(a)&&a.constructor===Object};Da=function(a,b,c){if(!Array.isArray(a)||a.length)return!1;const d=a[_.v]|0;if(d&1)return!0;if(!(b&&(Array.isArray(b)?b.includes(c):b.has(c))))return!1;_.wa(a,d|1);return!0};_.Ea=function(a){if(a&2)throw Error();};Ha=function(a,b){(b=_.Ga?b[_.Ga]:void 0)&&(a[_.Ga]=_.va(b))};_.Ja=function(){const a=Error();Ia(a,"incident");_.ba(a)};_.Ka=function(a){a=Error(a);Ia(a,"warning");return a};
_.Ma=function(a){if("boolean"!==typeof a)throw Error("r`"+_.La(a)+"`"+a);return a};_.Na=function(a){if(!Number.isFinite(a))throw _.Ka("enum");return a|0};_.Oa=function(a){if("number"!==typeof a)throw _.Ka("int32");if(!Number.isFinite(a))throw _.Ka("int32");return a|0};_.Pa=function(a){if(null!=a&&"string"!==typeof a)throw Error();return a};_.Qa=function(a){return null==a||"string"===typeof a?a:void 0};
_.Sa=function(a,b,c){if(null!=a&&"object"===typeof a&&a.Kd===_.Ra)return a;if(Array.isArray(a)){var d=a[_.v]|0,e=d;0===e&&(e|=c&32);e|=c&2;e!==d&&_.wa(a,e);return new b(a)}};_.Ua=function(a,b){Ta=b;a=new a(b);Ta=void 0;return a};
_.Va=function(a,b,c){null==a&&(a=Ta);Ta=void 0;if(null==a){var d=96;c?(a=[c],d|=512):a=[];b&&(d=d&-16760833|(b&1023)<<14)}else{if(!Array.isArray(a))throw Error("s");d=a[_.v]|0;if(d&2048)throw Error("t");if(d&64)return a;d|=64;if(c&&(d|=512,c!==a[0]))throw Error("u");a:{c=a;const e=c.length;if(e){const f=e-1;if(Ca(c[f])){d|=256;b=f-(+!!(d&512)-1);if(1024<=b)throw Error("v");d=d&-16760833|(b&1023)<<14;break a}}if(b){b=Math.max(b,e-(+!!(d&512)-1));if(1024<b)throw Error("w");d=d&-16760833|(b&1023)<<14}}}_.wa(a,
d);return a};Xa=function(a,b){return Wa(b)};Wa=function(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "boolean":return a?1:0;case "object":if(a)if(Array.isArray(a)){if(Da(a,void 0,0))return}else{if(_.ua(a))return _.ta(a);if("function"==typeof _.Ya&&a instanceof _.Ya)return a.j()}}return a};Za=function(a,b,c){const d=_.va(a);var e=d.length;const f=b&256?d[e-1]:void 0;e+=f?-1:0;for(b=b&512?1:0;b<e;b++)d[b]=c(d[b]);if(f){b=d[b]={};for(const g in f)b[g]=c(f[g])}Ha(d,a);return d};
ab=function(a,b,c,d,e){if(null!=a){if(Array.isArray(a))a=Da(a,void 0,0)?void 0:e&&(a[_.v]|0)&2?a:$a(a,b,c,void 0!==d,e);else if(Ca(a)){const f={};for(let g in a)f[g]=ab(a[g],b,c,d,e);a=f}else a=b(a,d);return a}};$a=function(a,b,c,d,e){const f=d||c?a[_.v]|0:0;d=d?!!(f&32):void 0;const g=_.va(a);for(let h=0;h<g.length;h++)g[h]=ab(g[h],b,c,d,e);c&&(Ha(g,a),c(f,g));return g};bb=function(a){return a.Kd===_.Ra?a.toJSON():Wa(a)};
cb=function(a,b,c=ya){if(null!=a){if(a instanceof Uint8Array)return b?a:new Uint8Array(a);if(Array.isArray(a)){var d=a[_.v]|0;if(d&2)return a;b&&(b=0===d||!!(d&32)&&!(d&64||!(d&16)));return b?_.wa(a,(d|34)&-12293):$a(a,cb,d&4?ya:c,!0,!0)}a.Kd===_.Ra&&(c=a.ma,d=c[_.v],a=d&2?a:_.Ua(a.constructor,_.db(c,d,!0)));return a}};_.db=function(a,b,c){const d=c||b&2?ya:xa,e=!!(b&32);a=Za(a,b,f=>cb(f,e,d));a[_.v]=a[_.v]|32|(c?2:0);return a};
_.eb=function(a){const b=a.ma,c=b[_.v];return c&2?_.Ua(a.constructor,_.db(b,c,!1)):a};fb=function(a,b,c,d){b=d+(+!!(b&512)-1);if(!(0>b||b>=a.length||b>=c))return a[b]};_.gb=function(a,b,c,d,e){const f=za(b);if(c>=f||e){let g=b;if(b&256)e=a[a.length-1];else{if(null==d)return g;e=a[f+(+!!(b&512)-1)]={};g|=256}e[c]=d;c<f&&(a[c+(+!!(b&512)-1)]=void 0);g!==b&&_.wa(a,g);return g}a[c+(+!!(b&512)-1)]=d;b&256&&(a=a[a.length-1],c in a&&delete a[c]);return b};
_.ib=function(a,b,c,d){a=a.ma;let e=a[_.v];const f=_.hb(a,e,c,d);b=_.Sa(f,b,e);b!==f&&null!=b&&_.gb(a,e,c,b,d);return b};_.jb=function(a,b){return null!=a?a:b};
lb=function(a,b,c){var d=kb?void 0:a.constructor.oa;const e=(c?a.ma:b)[_.v];a=b.length;if(!a)return b;let f,g;if(Ca(c=b[a-1])){a:{var h=c;let n={},p=!1;for(var k in h){let r=h[k];if(Array.isArray(r)){let t=r;if(Da(r,d,+k)||Ba(r)&&0===r.size)r=null;r!=t&&(p=!0)}null!=r?n[k]=r:p=!0}if(p){for(var l in n){h=n;break a}h=null}}h!=c&&(f=!0);a--}for(k=+!!(e&512)-1;0<a;a--){l=a-1;c=b[l];l-=k;if(!(null==c||Da(c,d,l)||Ba(c)&&0===c.size))break;g=!0}if(!f&&!g)return b;b=Array.prototype.slice.call(b,0,a);h&&b.push(h);
return b};_.w=function(a,b){return null!=a?!!a:!!b};_.x=function(a,b){void 0==b&&(b="");return null!=a?a:b};_.mb=function(a){for(const b in a)return!1;return!0};nb="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
ob=function(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("a");};pb=ob(this);qb=function(a,b){if(b)a:{var c=pb;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&null!=b&&nb(c,a,{configurable:!0,writable:!0,value:b})}};qb("globalThis",function(a){return a||pb});
qb("Promise.prototype.finally",function(a){return a?a:function(b){return this.then(function(c){return Promise.resolve(b()).then(function(){return c})},function(c){return Promise.resolve(b()).then(function(){throw c;})})}});var tb,ub,wb,xb;_.rb=_.rb||{};_.q=this||self;tb=function(a){var b=_.sb("WIZ_global_data.oxN3nb");a=b&&b[a];return null!=a?a:!1};ub=_.q._F_toggles||[];_.sb=function(a,b){a=a.split(".");b=b||_.q;for(var c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b};_.La=function(a){var b=typeof a;return"object"!=b?b:a?Array.isArray(a)?"array":b:"null"};_.vb="closure_uid_"+(1E9*Math.random()>>>0);wb=function(a,b,c){return a.call.apply(a.bind,arguments)};
xb=function(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}};_.y=function(a,b,c){_.y=Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?wb:xb;return _.y.apply(null,arguments)};
_.z=function(a,b){a=a.split(".");var c=_.q;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===b?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b};_.A=function(a,b){function c(){}c.prototype=b.prototype;a.W=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.hj=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}};_.A(_.aa,Error);_.aa.prototype.name="CustomError";_.yb=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};var zb=!!(ub[0]&128),Ab=!!(ub[0]&256),Bb=!!(ub[0]&2);var da=zb?Ab:tb(610401301),kb=zb?Bb:tb(188588736);var ea,Cb=_.q.navigator;ea=Cb?Cb.userAgentData||null:null;_.ra=function(a,b){return Array.prototype.indexOf.call(a,b,void 0)};_.Db=function(a,b,c){Array.prototype.forEach.call(a,b,c)};_.Eb=function(a){_.Eb[" "](a);return a};_.Eb[" "]=function(){};var Rb,Sb,Xb;_.Fb=_.ia();_.B=_.ja();_.Gb=_.u("Edge");_.Hb=_.Gb||_.B;_.Ib=_.u("Gecko")&&!(-1!=_.ca().toLowerCase().indexOf("webkit")&&!_.u("Edge"))&&!(_.u("Trident")||_.u("MSIE"))&&!_.u("Edge");_.Jb=-1!=_.ca().toLowerCase().indexOf("webkit")&&!_.u("Edge");_.Kb=_.qa();_.Lb=na()?"Windows"===ea.platform:_.u("Windows");_.Mb=na()?"Android"===ea.platform:_.u("Android");_.Nb=oa();_.Ob=_.u("iPad");_.Pb=_.u("iPod");_.Qb=_.pa();Rb=function(){var a=_.q.document;return a?a.documentMode:void 0};
a:{var Tb="",Ub=function(){var a=_.ca();if(_.Ib)return/rv:([^\);]+)(\)|;)/.exec(a);if(_.Gb)return/Edge\/([\d\.]+)/.exec(a);if(_.B)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(_.Jb)return/WebKit\/(\S+)/.exec(a);if(_.Fb)return/(?:Version)[ \/]?(\S+)/.exec(a)}();Ub&&(Tb=Ub?Ub[1]:"");if(_.B){var Vb=Rb();if(null!=Vb&&Vb>parseFloat(Tb)){Sb=String(Vb);break a}}Sb=Tb}_.Wb=Sb;if(_.q.document&&_.B){var Zb=Rb();Xb=Zb?Zb:parseInt(_.Wb,10)||void 0}else Xb=void 0;_.$b=Xb;_.ac=_.ka();_.bc=oa()||_.u("iPod");_.cc=_.u("iPad");_.dc=_.u("Android")&&!(la()||_.ka()||_.ia()||_.u("Silk"));_.ec=la();_.fc=_.ma()&&!_.pa();_.gc="undefined"!==typeof TextDecoder;_.hc="undefined"!==typeof TextEncoder;var ic;_.v=Symbol();ic=Symbol();_.jc=Symbol();_.wa=(a,b)=>{a[_.v]=b;return a};var Aa,kc,mc,nc,pc;_.Ra={};Aa={};mc=[];_.wa(mc,55);_.lc=Object.freeze(mc);nc=class{};pc=class{};Object.freeze(new nc);Object.freeze(new pc);var Ia=function(a,b){a.__closure__error__context__984382||(a.__closure__error__context__984382={});a.__closure__error__context__984382.severity=b};var qc;var Ta;_.rc=function(a,b){a=a.ma;return _.hb(a,a[_.v],b)};_.hb=function(a,b,c,d){if(-1===c)return null;const e=za(b);if(c>=e){if(b&256)return a[a.length-1][c]}else{var f=a.length;if(d&&b&256&&(d=a[f-1][c],null!=d)){if(fb(a,b,e,c)&&null!=ic){var g;a=null!=(g=qc)?g:qc={};g=a[ic]||0;4<=g||(a[ic]=g+1,_.Ja())}return d}return fb(a,b,e,c)}};_.sc=function(a,b,c){const d=a.ma;let e=d[_.v];_.Ea(e);_.gb(d,e,b,c);return a};
_.D=function(a,b){a=_.rc(a,b);return null==a||"boolean"===typeof a?a:"number"===typeof a?!!a:void 0};_.E=function(a,b,c,d=!1){b=_.ib(a,b,c,d);if(null==b)return b;a=a.ma;let e=a[_.v];if(!(e&2)){const f=_.eb(b);f!==b&&(b=f,_.gb(a,e,c,b,d))}return b};_.F=function(a,b,c){null==c&&(c=void 0);return _.sc(a,b,c)};_.G=function(a,b){return _.Qa(_.rc(a,b))};_.I=function(a,b,c=!1){return _.jb(_.D(a,b),c)};_.J=function(a,b){return _.jb(_.G(a,b),"")};_.K=function(a,b,c){return _.sc(a,b,null==c?c:_.Ma(c))};
_.L=function(a,b,c){return _.sc(a,b,null==c?c:_.Oa(c))};_.M=function(a,b,c){return _.sc(a,b,_.Pa(c))};_.N=function(a,b,c){return _.sc(a,b,null==c?c:_.Na(c))};_.Q=class{constructor(a,b,c){this.ma=_.Va(a,b,c)}toJSON(){return kc?lb(this,this.ma,!1):lb(this,$a(this.ma,bb,void 0,void 0,!1),!0)}Fa(){kc=!0;try{return JSON.stringify(this.toJSON(),Xa)}finally{kc=!1}}wc(){return!!((this.ma[_.v]|0)&2)}};_.Q.prototype.Kd=_.Ra;_.Q.prototype.toString=function(){return lb(this,this.ma,!1).toString()};_.tc=Symbol();_.uc=Symbol();_.vc=Symbol();_.wc=Symbol();_.xc=Symbol();var yc=class extends _.Q{constructor(){super()}};_.zc=class extends _.Q{constructor(){super()}C(a){return _.L(this,3,a)}};var Ac=class extends _.Q{constructor(a){super(a)}Nc(a){return _.M(this,24,a)}};_.Bc=class extends _.Q{constructor(a){super(a)}};_.Cc=function(){this.Da=this.Da;this.ha=this.ha};_.Cc.prototype.Da=!1;_.Cc.prototype.isDisposed=function(){return this.Da};_.Cc.prototype.fa=function(){this.Da||(this.Da=!0,this.P())};_.Cc.prototype.P=function(){if(this.ha)for(;this.ha.length;)this.ha.shift()()};var Dc=class extends _.Cc{constructor(){var a=window;super();this.o=a;this.i=[];this.j={}}resolve(a){var b=this.o;a=a.split(".");for(var c=a.length,d=0;d<c;++d)if(b[a[d]])b=b[a[d]];else return null;return b instanceof Function?b:null}Cb(){for(var a=this.i.length,b=this.i,c=[],d=0;d<a;++d){var e=b[d].i(),f=this.resolve(e);if(f&&f!=this.j[e])try{b[d].Cb(f)}catch(g){}else c.push(b[d])}this.i=c.concat(b.slice(a))}};var Fc=class extends _.Cc{constructor(){var a=_.Ec;super();this.o=a;this.A=this.i=null;this.v=0;this.B={};this.j=!1;a=window.navigator.userAgent;0<=a.indexOf("MSIE")&&0<=a.indexOf("Trident")&&(a=/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a))&&a[1]&&9>parseFloat(a[1])&&(this.j=!0)}C(a,b){this.i=b;this.A=a;b.preventDefault?b.preventDefault():b.returnValue=!1}};_.Gc=class extends _.Q{constructor(a){super(a)}};var Hc=class extends _.Q{constructor(a){super(a)}};var Kc;_.Ic=function(a,b){if(a.i){const c=new yc;_.M(c,1,b.message);_.M(c,2,b.stack);_.L(c,3,b.lineNumber);_.N(c,5,1);b=new _.zc;_.F(b,40,c);a.i.log(98,b)}};Kc=class{constructor(){var a=Jc;this.i=null;_.I(a,4,!0)}log(a){_.Ic(this,a)}};var Mc;_.Lc=function(a){if(0<a.o.length){var b=void 0!==a.i,c=void 0!==a.j;if(b||c){b=b?a.v:a.A;c=a.o;a.o=[];try{_.Db(c,b,a)}catch(d){console.error(d)}}}};_.Nc=class{constructor(a){this.i=a;this.j=void 0;this.o=[]}then(a,b,c){this.o.push(new Mc(a,b,c));_.Lc(this)}resolve(a){if(void 0!==this.i||void 0!==this.j)throw Error("A");this.i=a;_.Lc(this)}v(a){a.j&&a.j.call(a.i,this.i)}A(a){a.o&&a.o.call(a.i,this.j)}};Mc=class{constructor(a,b,c){this.j=a;this.o=b;this.i=c}};_.Oc=a=>{var b="uc";if(a.uc&&a.hasOwnProperty(b))return a.uc;b=new a;return a.uc=b};_.Pc=class{constructor(){this.v=new _.Nc;this.i=new _.Nc;this.D=new _.Nc;this.B=new _.Nc;this.C=new _.Nc;this.A=new _.Nc;this.o=new _.Nc;this.j=new _.Nc;this.F=new _.Nc}ha(){return this.v}M(){return this.i}N(){return this.D}L(){return this.B}Da(){return this.C}K(){return this.A}J(){return this.o}G(){return this.j}static i(){return _.Oc(_.Pc)}};var Uc;_.Rc=function(){return _.E(_.Qc,Ac,1)};_.Sc=function(){return _.E(_.Qc,_.Bc,5)};Uc=class extends _.Q{constructor(){super(Tc)}};var Tc;window.gbar_&&window.gbar_.CONFIG?Tc=window.gbar_.CONFIG[0]||{}:Tc=[];_.Qc=new Uc;var Jc=_.E(_.Qc,Hc,3)||new Hc;_.Rc()||new Ac;_.Ec=new Kc;_.z("gbar_._DumpException",function(a){_.Ec?_.Ec.log(a):console.error(a)});_.Vc=new Fc;var Xc;_.Yc=function(a,b){var c=_.Wc.i();if(a in c.i){if(c.i[a]!=b)throw new Xc;}else{c.i[a]=b;const h=c.j[a];if(h)for(let k=0,l=h.length;k<l;k++){b=h[k];var d=c.i;delete b.i[a];if(_.mb(b.i)){for(var e=b.j.length,f=Array(e),g=0;g<e;g++)f[g]=d[b.j[g]];b.o.apply(b.v,f)}}delete c.j[a]}};_.Wc=class{constructor(){this.i={};this.j={}}static i(){return _.Oc(_.Wc)}};_.Zc=class extends _.aa{constructor(){super()}};Xc=class extends _.Zc{};_.z("gbar.A",_.Nc);_.Nc.prototype.aa=_.Nc.prototype.then;_.z("gbar.B",_.Pc);_.Pc.prototype.ba=_.Pc.prototype.M;_.Pc.prototype.bb=_.Pc.prototype.N;_.Pc.prototype.bd=_.Pc.prototype.Da;_.Pc.prototype.bf=_.Pc.prototype.ha;_.Pc.prototype.bg=_.Pc.prototype.L;_.Pc.prototype.bh=_.Pc.prototype.K;_.Pc.prototype.bj=_.Pc.prototype.J;_.Pc.prototype.bk=_.Pc.prototype.G;_.z("gbar.a",_.Pc.i());window.gbar&&window.gbar.ap&&window.gbar.ap(window.gbar.a);var $c=new Dc;_.Yc("api",$c);
var ad=_.Sc()||new _.Bc,bd=window,cd=_.x(_.G(ad,8));bd.__PVT=cd;_.Yc("eq",_.Vc);
}catch(e){_._DumpException(e)}
try{
_.dd=class extends _.Q{constructor(a){super(a)}};
}catch(e){_._DumpException(e)}
try{
var ed=class extends _.Q{constructor(){super()}};var fd=class extends _.Cc{constructor(){super();this.j=[];this.i=[]}o(a,b){this.j.push({features:a,options:b})}init(a,b,c){window.gapi={};var d=window.___jsl={};d.h=_.x(_.G(a,1));null!=_.D(a,12)&&(d.dpo=_.w(_.I(a,12)));d.ms=_.x(_.G(a,2));d.m=_.x(_.G(a,3));d.l=[];_.J(b,1)&&(a=_.G(b,3))&&this.i.push(a);_.J(c,1)&&(c=_.G(c,2))&&this.i.push(c);_.z("gapi.load",(0,_.y)(this.o,this));return this}};var gd=_.E(_.Qc,_.Gc,14);if(gd){var hd=_.E(_.Qc,_.dd,9)||new _.dd,id=new ed,jd=new fd;jd.init(gd,hd,id);_.Yc("gs",jd)};
}catch(e){_._DumpException(e)}
})(this.gbar_);
// Google Inc.
</script><script nonce="">try {const preferences = JSON.parse(window.localStorage.getItem("datalab_prefs_luckychamp141@gmail.com")); document.querySelector('html') .setAttribute('theme', preferences['siteTheme'] || 'default');} catch (e) {}</script><script nonce="">window.performance.mark('head_start');</script><link rel="stylesheet" href="./untitled10_files/bundle.css"><script src="./untitled10_files/common_webcomponentsjs_webcomponents-lite.js.download" nonce=""></script><script nonce="">var colabVersionTag = 'colab_20240423-060158_RC00_627338680'; var colabScsVersion = 'bb50aebe8935a785f12380a66727b2c4'; var hl = 'en'; var colabExperiments = JSON.parse('\x7b\x22ai_unsubscribed_warning\x22: true, \x22aida_cell_button\x22: false, \x22aida_complete_code_model_id\x22: \x22\x22, \x22aida_do_conversation_model_id\x22: \x22\x22, \x22aida_generate_code_model_id\x22: \x22\x22, \x22aida_in_editor\x22: false, \x22allowed_public_url_domains\x22: \x5b\x22huggingface.co\x22, \x22dagshub.com\x22, \x22storage.googleapis.com\x22\x5d, \x22auto_execute_prereqs\x22: false, \x22backend_version\x22: \x22next\x22, \x22bq_tid\x22: true, \x22cell_tags\x22: false, \x22chat\x22: true, \x22classified_generate\x22: false, \x22classroom_iframe_parent_origin\x22: \x22\x22, \x22client_text_compose\x22: false, \x22client_trim_completion_text\x22: 100, \x22cloud_origin\x22: \x22\x22, \x22commands_in_toolbar\x22: false, \x22comment_poll_long\x22: 900000, \x22comment_poll_short\x22: 60000, \x22completion_throttle_millis\x22: 300, \x22compose_skip_suffix_check\x22: false, \x22converse_temp\x22: \x22\x22, \x22crawler\x22: false, \x22critique_comments\x22: false, \x22dbu\x22: \x22\x22, \x22debug_external\x22: \x22external\x22, \x22debug_prod\x22: \x22prod\x22, \x22dep_cells_commands\x22: false, \x22development\x22: false, \x22document_change_poll_interval\x22: 30000, \x22drive_anon_api_key\x22: \x22AIzaSyB10s2vWUTwP0pj20wZoxmpZIt3rRodYeg\x22, \x22drive_api_key\x22: \x22AIzaSyCN_sSPJMpYrAzC5AtTrltNC8oRmLtoqBk\x22, \x22drive_background_save_project_number\x22: \x22948411933973\x22, \x22drive_realtime_project_number\x22: \x22\x22, \x22embedded_connection_poll\x22: false, \x22embedding_app\x22: \x22\x22, \x22enable_adhoc_backends\x22: false, \x22enable_more_reprs\x22: true, \x22explain_cell\x22: false, \x22explain_error\x22: true, \x22explain_error_prompt_next\x22: true, \x22explain_error_trim_traceback\x22: true, \x22external_trusted_github_org_repos_quick_add\x22: \x22GoogleChrome\/CrUX,google\/generative-ai-docs\x22, \x22filter_repetitive_suggestions\x22: false, \x22first_party_auth\x22: true, \x22fix_imports\x22: false, \x22generate_code\x22: true, \x22generate_code_prompt\x22: true, \x22generate_df\x22: true, \x22generate_fix\x22: true, \x22generate_temp\x22: \x22\x22, \x22get_started\x22: false, \x22gis_auth\x22: true, \x22github_client_id\x22: \x225036cf6d81e65aaa6340\x22, \x22gpu_utilization_check_interval_ms\x22: 600000, \x22hats_surveys\x22: true, \x22hrc\x22: false, \x22import_data\x22: false, \x22internal_chat\x22: false, \x22internal_schedule\x22: false, \x22is_prober\x22: false, \x22jsraw\x22: \x22compiled\x22, \x22key_promoter\x22: false, \x22kr\x22: false, \x22local_cloud_apis\x22: false, \x22local_service_worker\x22: false, \x22log_hover_type\x22: false, \x22lsp_diagnostic_threshold_b312769838\x22: 0, \x22lsp_diagnostics_reporting\x22: false, \x22lsp_inlay_hint\x22: false, \x22lsrp\x22: 0, \x22ml_banner\x22: false, \x22ml_enabled\x22: true, \x22mlpp\x22: false, \x22mlpp_multiline\x22: false, \x22mlpp_on_by_default\x22: true, \x22mlpp_trim_completion_text\x22: 100, \x22mobile\x22: false, \x22next_steps\x22: true, \x22nl2code_missing_imports\x22: true, \x22no_fun\x22: false, \x22og_dark\x22: true, \x22outage_notification\x22: \x22\x22, \x22outage_notification_link\x22: \x22\x22, \x22output_menu\x22: false, \x22outputframe_version\x22: \x22\x22, \x22override_suf_params_for_test\x22: false, \x22quickchart_button\x22: true, \x22recaptcha_polling_frequency_ms\x22: 300000, \x22recaptcha_v2_site_key\x22: \x226LfQttQUAAAAADuPanA_VZMaZgBAOnHZNuuqUewp\x22, \x22recaptcha_v3_site_key\x22: \x226LfQPtEUAAAAAHBpAdFng54jyuB1V5w5dofknpip\x22, \x22reconnect_max_delay_seconds\x22: 300, \x22replace_indented_block_with_blockquote\x22: true, \x22require_ai_consent\x22: true, \x22resource_poll_interval_ms\x22: 10000, \x22rp_kxhr\x22: false, \x22rp_lsp\x22: false, \x22rp_serve_kernel_port\x22: false, \x22rp_term\x22: false, \x22rp_token_refresh_headroom_millis\x22: 300000, \x22rt\x22: false, \x22runtime_env_overrides\x22: \x22\\n          \x5b\\n            \x5b\\\x22ENABLE_DIRECTORYPREFETCHER\\\x22, \\\x221\\\x22\x5d\\n          \x5d\\n        \x22, \x22runtime_type_for_test\x22: \x22\x22, \x22server_execution_queue\x22: true, \x22session_resume_coalesce\x22: true, \x22sheets_paste\x22: true, \x22show_free_quota_info\x22: true, \x22show_internal_ai_warning\x22: false, \x22show_payments_interstitial\x22: false, \x22show_relnotes_on_open\x22: true, \x22show_signup_survey\x22: true, \x22show_subscription_renewal_time\x22: false, \x22show_switch_to_prod_link\x22: false, \x22sql_cell\x22: false, \x22sql_cell_buttons\x22: false, \x22storage_partition_trial\x22: true, \x22storage_partition_trial_token\x22: \x22AmUpB2+Hlwk73pYiEMbnkef\/dprJi1I9rClec33apyFsbVOaCIRN29Rk9M4ht5Otgbp+thCc3MMD73GyCNfEWAkAAAB3eyJvcmlnaW4iOiJodHRwczovL2NvbGFiLnJlc2VhcmNoLmdvb2dsZS5jb206NDQzIiwiZmVhdHVyZSI6IkRpc2FibGVUaGlyZFBhcnR5U3RvcmFnZVBhcnRpdGlvbmluZyIsImV4cGlyeSI6MTcyNTQwNzk5OX0\x3d\x22, \x22suspicious_code_matches\x22: \x22\x22, \x22suspicious_code_regexs\x22: \x22\x22, \x22term4all\x22: false, \x22text_compose_report_changes_millis\x22: 10000, \x22text_span_comments\x22: false, \x22tpu_vm\x22: true, \x22unified_compose\x22: true, \x22unmanaged_vm_min_label_block\x22: \x22\x22, \x22unmanaged_vm_min_label_warn\x22: \x22\x22, \x22unmanaged_vm_min_release_tag_block\x22: \x22\x22, \x22unmanaged_vm_min_release_tag_warn\x22: \x22\x22, \x22use_corplogin\x22: true, \x22use_dm_sql_lsp\x22: false, \x22use_upticks_for_internalai\x22: false, \x22user_visible_gpu_types\x22: \x5b\x22T4\x22, \x22V100\x22, \x22A100\x22, \x22L4\x22\x5d, \x22uxr_survey\x22: false, \x22verbose_warnings\x22: false, \x22vertex_ai_api_environment_override\x22: \x22\x22, \x22waffle\x22: false, \x22workstations\x22: false, \x22ids\x22: \x5b20730186, 20730177, 20730159, 20730150, 20730183, 20730189, 20730129\x5d, \x22flag_ids\x22: \x7b\x22ai_unsubscribed_warning\x22: 45504730, \x22aida_cell_button\x22: 45425510, \x22aida_complete_code_model_id\x22: 45427660, \x22aida_do_conversation_model_id\x22: 45427664, \x22aida_generate_code_model_id\x22: 45427663, \x22aida_in_editor\x22: 45425507, \x22allowed_public_url_domains\x22: 45425558, \x22auto_execute_prereqs\x22: 45624305, \x22backend_version\x22: 45425541, \x22bq_tid\x22: 45425617, \x22cell_tags\x22: 45425779, \x22chat\x22: 45425490, \x22classified_generate\x22: 45425499, \x22classroom_iframe_parent_origin\x22: 45425537, \x22client_text_compose\x22: 45425512, \x22client_trim_completion_text\x22: 45425628, \x22cloud_origin\x22: 45425538, \x22commands_in_toolbar\x22: 45425502, \x22comment_poll_long\x22: 45425588, \x22comment_poll_short\x22: 45425587, \x22completion_throttle_millis\x22: 45621713, \x22compose_skip_suffix_check\x22: 45615470, \x22converse_temp\x22: 45425509, \x22crawler\x22: 45425491, \x22critique_comments\x22: 45612076, \x22dbu\x22: 45425545, \x22debug_external\x22: 45425470, \x22debug_prod\x22: 45425471, \x22dep_cells_commands\x22: 45622249, \x22development\x22: 45425544, \x22document_change_poll_interval\x22: 45425589, \x22drive_anon_api_key\x22: 45425478, \x22drive_api_key\x22: 45425473, \x22drive_background_save_project_number\x22: 45425479, \x22drive_realtime_project_number\x22: 45425629, \x22embedded_connection_poll\x22: 45491618, \x22enable_adhoc_backends\x22: 45425506, \x22enable_more_reprs\x22: 45613354, \x22explain_cell\x22: 45425505, \x22explain_error\x22: 45425487, \x22explain_error_prompt_next\x22: 45615229, \x22explain_error_trim_traceback\x22: 45618831, \x22external_trusted_github_org_repos_quick_add\x22: 45425555, \x22filter_repetitive_suggestions\x22: 45615781, \x22first_party_auth\x22: 45425560, \x22fix_imports\x22: 45624140, \x22generate_code\x22: 45425492, \x22generate_code_prompt\x22: 45425488, \x22generate_df\x22: 45425503, \x22generate_fix\x22: 45425504, \x22generate_temp\x22: 45425508, \x22get_started\x22: 45430267, \x22gis_auth\x22: 45425625, \x22github_client_id\x22: 45425556, \x22gpu_utilization_check_interval_ms\x22: 45425561, \x22hats_surveys\x22: 45425559, \x22import_data\x22: 45430411, \x22internal_chat\x22: 45622872, \x22internal_schedule\x22: 45425578, \x22is_prober\x22: 45429104, \x22jsraw\x22: 45425557, \x22key_promoter\x22: 45425570, \x22local_cloud_apis\x22: 45425630, \x22local_service_worker\x22: 45425550, \x22log_hover_type\x22: 45425602, \x22lsp_diagnostic_threshold_b312769838\x22: 45618432, \x22lsp_diagnostics_reporting\x22: 45425604, \x22lsp_inlay_hint\x22: 45614695, \x22lsrp\x22: 45425612, \x22ml_banner\x22: 45425631, \x22ml_enabled\x22: 45425493, \x22mlpp\x22: 45425608, \x22mlpp_multiline\x22: 45425623, \x22mlpp_on_by_default\x22: 45425609, \x22mlpp_trim_completion_text\x22: 45425622, \x22mobile\x22: 45425562, \x22next_steps\x22: 45428239, \x22nl2code_missing_imports\x22: 45615676, \x22no_fun\x22: 45425540, \x22og_dark\x22: 45459627, \x22outage_notification\x22: 45425584, \x22outage_notification_link\x22: 45425585, \x22output_menu\x22: 45617054, \x22outputframe_version\x22: 45425591, \x22override_suf_params_for_test\x22: 45425592, \x22quickchart_button\x22: 45425501, \x22recaptcha_polling_frequency_ms\x22: 45425582, \x22recaptcha_v2_site_key\x22: 45425581, \x22recaptcha_v3_site_key\x22: 45425580, \x22reconnect_max_delay_seconds\x22: 45425539, \x22replace_indented_block_with_blockquote\x22: 45623563, \x22require_ai_consent\x22: 45425489, \x22resource_poll_interval_ms\x22: 45425590, \x22rp_kxhr\x22: 45491686, \x22rp_lsp\x22: 45462965, \x22rp_serve_kernel_port\x22: 45572083, \x22rp_term\x22: 45426219, \x22rp_token_refresh_headroom_millis\x22: 45517773, \x22rt\x22: 45425624, \x22runtime_env_overrides\x22: 45425583, \x22runtime_type_for_test\x22: 45425586, \x22server_execution_queue\x22: 45425600, \x22session_resume_coalesce\x22: 45425603, \x22sheets_paste\x22: 45428502, \x22show_free_quota_info\x22: 45620295, \x22show_internal_ai_warning\x22: 45622780, \x22show_payments_interstitial\x22: 45425543, \x22show_relnotes_on_open\x22: 45428128, \x22show_signup_survey\x22: 45425620, \x22show_subscription_renewal_time\x22: 45425569, \x22show_switch_to_prod_link\x22: 45425483, \x22sql_cell\x22: 45425497, \x22sql_cell_buttons\x22: 45425498, \x22suspicious_code_matches\x22: 45425615, \x22suspicious_code_regexs\x22: 45425616, \x22term4all\x22: 45425542, \x22text_compose_report_changes_millis\x22: 45425568, \x22text_span_comments\x22: 45545873, \x22tpu_vm\x22: 45614812, \x22unified_compose\x22: 45425627, \x22unmanaged_vm_min_label_block\x22: 45425546, \x22unmanaged_vm_min_label_warn\x22: 45425547, \x22unmanaged_vm_min_release_tag_block\x22: 45425548, \x22unmanaged_vm_min_release_tag_warn\x22: 45425549, \x22use_corplogin\x22: 45425606, \x22use_dm_sql_lsp\x22: 45425610, \x22use_upticks_for_internalai\x22: 45629267, \x22user_visible_gpu_types\x22: 45620529, \x22uxr_survey\x22: 45425618, \x22verbose_warnings\x22: 45425551, \x22vertex_ai_api_environment_override\x22: 45612077, \x22waffle\x22: 45446491, \x22workstations\x22: 45425626\x7d\x7d'); var colabUserEmail = 'luckychamp141@gmail.com'; var colabRenderDataToken = 'AFWLbD0-OuIyhI3N_6mOo9GhuQlFZDhLKn9Rh2Bt7VV3W2sxPbCcOLpU2W22vPvg7PIySxt0QzRHzXXaK0hrrvLZpZ-R6_dbJBnm'; var colabConfig = '\x5b\x5b\x22luckychamp141@gmail.com\x22,\x5b1,\x22AHXL0D3qJfCn5V0nxTHcdSo25sLOCvbHzYM5RmBlskixw4QUTEL7L4oAk89rvXCa7d0LUFhaojXqpyVac8JLJfe74k+GfUjbx2YQIW4GrMzGpygW8x5pJghhZgCsmW6VWFL4X90c7ELbX0v7IBNsLNgjYregeLCmd8FaWCvHxG\/+elaYGc24hikGb3A2gAntOO5WxIW8kSNvyxGDVt+pQ0Ak82kwIrsGFG4ggV32wiJVaWhpQavb4C3yHAw6on06X99RDVnbSpd4zU5SJLhnA2b3HoeoPnJuSDhkHFeRJH3YM2QACgZCAHG9b5f9\/4HLD\/+2aUEw2xQNb+lDDOshlYBsFpGcP2vCnKfIJDNA+9GGB+T\/tHDncobBnBcmDJ3OmrBfI4zf9UQcygg4zjZ1zHWBS83bRBLY\/BPZ5+qMyJJCgfBGEYss+13\/jw8PEpcLQQEqohEP+YrxPw+fk7UeYYLXtHdm3OQmXIWSEzAEHZY0vxEXVlyRXLdJ96gA9N\/RydAUYiPcaNDEeOJCNyclrvRdeFUxZAGUUJvPYfI9NZOznHxEZ4NBIChILVgt3SdC+Zhbbu5XhpjbhBmFTsjD5k1CoiF5cW0aidIwYAalyfFmG2qJ6VuhSdETpL9JmiivcepdC2cRE7pO1H1aznOSwIqnl2CYP4SHwYqePiOpkAygDnRLw0maIK1xrY+9ceeqUrt+lEuxp2cxSk9aoHp+9wqGBVz7\/MnN\/nVeXoweUGI4sQHwwHJpo2U1WGKuiKG\/\/mpQ\/sJMk+Cnyjol+NTNPwWmznozW1Rk3xKerd9kj7PfGlAUlcsIE94favA+rBaAVTzHVrQXMAHGtLF3S3wfbf58tGGs41UWaJgqblxJ5UuSWTnMAU3ZOwfE1gkP15oc3Oux+WSJpCtzCVrc4pvjw64IxurFv7sm9rpWDPN4msEU+S5kHAHiwNLEM+SSJ7VzM0WWgm\/qw9Yo24ayS0+QPlFpOx+QtJqO\/uDsWEM38OA+bfpVLL4s6RZcFQ\/6jNanUzAo8eEex4fjbdh1mNTNYpU6O9b\/2KZUelbBSqnevN5WA46F71J6MoUq\/iG2L65C6qvVVCG\/obKQgyo\/caU2zmQZ0Mu96115nCnVMFdIxnS+EsZmAuhJ5b0ttASWtHE5KDMBJT7fR797+zzAd7u4I+dAsJ19YfRE3ysVeZz3KDUjMmdLeTUb20wKGlR5nY3K8vjCE3wIxW0LtoyEDrTsA9w+PXp+zSuhPQpE43ABLTbu\/VMdaK021\/VwgmhYAykJ8sS3tH+ORFO13KonfC88Z5ZOcF528ozXw+A4AV\/W51MGppQeaVuHdzllJ\/jM6Djp5oLLRLnWZzHKbcNqhdoZvbIGUlWEFp6rPIrTsjoYl\/ElHz5ibBV+DP6SlHuivbOJ8qaTTNEJ0xEqSAXboc1d3iXRWm4fijXDDSYPRZhK6u3lSbU7S0J9m3P+3cnfFL9Yhr2N1mFcCohTb0j5\/bhCgCrZTyi\/K9dwuMtXXjXjoytNExy4hvb3HzjlfowCIOJmofsks7B8T3MiIB2su46V0w7aZ6uwHxHDstNlin8XAWdSigVPuP\/ZaPuAoEK64Imr+YdtyKB+rQLkJINTUtjzJmZf9Ne+cxpsBF325Vrofc0+\/n3a\/FKIsJ\/6DcBkvAbZ6ccpeXUKDKovad5IoUIusj5O2gFKbWT5wLJy01Q0Z08JGMfnxncK4STw5\/2293YZNEfo1IbbWxyYZVW096Ia8URMUaeejqK5wYZT+a97cE0LevBiKUnoNhmVZAjJlcf6VOD7E5KKfMDZcCZpAtFX9w0xvUPTlwStAojnibyn+u9khPyrovsIBZVYuKlgciRafg428rA2hzT5KNltIx3kZsrKPw1vNKwcRq7RbF4OLismPvCDAIRao38\x3d\x22,\x22AJ9oCCxUXfTQ1YtCvXo5uwCLm6FOZ4d61fFWeBiEb9M1S3RuFqe42cmfgNhp4dI+Cs6COBHLaoPpX7ADouIFZWInLcJb39AzvuWaYXv0BHT\/8Y4V92W1GgrYwdYlayTHFKklOUV5jz2s\x22,\x22https:\/\/payments.google.com\/payments\x22,0,0,null,null,null,\x22$9.99\x22,\x22$49.99\x22,\x22$9.99\x22,\x22$49.99\x22\x5d,\x5b1\x5d,\x22US\x22\x5d\x5d';</script><link id="favicon-link" rel="shortcut icon" href="https://ssl.gstatic.com/colaboratory-static/common/bb50aebe8935a785f12380a66727b2c4/img%2Ffavicon.ico"><meta name="google-site-verification" content="wRgpUU3mIRZPD1GORBPNonaotM72092B_DsqQFWNa4s"><meta name="google-site-verification" content="f5qdvL6RAXlBgHezvCLpPtvx2wU5ZgIzzPULroprlnA"><meta name="google-site-verification" content="-wL8iYJTC7X0zF9qBNDQUAd-P1ZkQUK-OhSgv4Wkf1M"><meta name="google-site-verification" content="o-EECwEDQeUpZv0jTmoGfCDX7dUI8Kul4ESepXmDnO0"><meta name="google-site-verification" content="sNOroO9gXrazN-oMODOm0Bs0_vw1R5QwZ-BfrSHn8Io"><meta name="google-site-verification" content="K1UdZBHJXQYnJGXIK1KuutmVy6dn3vG2sEyV9D1C6dM"><meta name="google-site-verification" content="wdGthzzfu0IjM3qpFqTuQL9poAQZAvAaFKyuzetLpIM"><meta name="google-site-verification" content="qZJ77guHGO0TObHUBRYVdXQlIhXBBuz8dahJVmIlzCo"><meta name="google-site-verification" content="7ahoeOOKT2ZR781GZ5xK4L9t7yO1ZOHc-gaoUALEYgw"><meta name="google-site-verification" content="PHgaSKwdxZELS21aixtLhfpvaHtKen9pnVJ25EI35Zs"><meta name="google-site-verification" content="Ozey1ptWUQW13_lCEhpPMOcmRBLqdyB3WEL-TJUjskU"><meta name="google-site-verification" content="rF5iXzWe9KZXJes1dQNhOUkS4_z_e97IrsVoCx2trek"><meta name="google-site-verification" content="cpB5oulaGwqSxsg4E-9q2MVbK87iE9NAUUVxdveucPw"><meta name="google-site-verification" content="8P-D5fVWgUIhw8X2BxnKJbf5itK0zxX0QhoBjbJFTe8"><meta name="google-site-verification" content="88fgsZDoVRBuRnDPMIEjcHCxsEXzODOqEsJoqtvQsDc"><meta name="google-site-verification" content="26aKGBCw3XblB5Ou01UhxY5WDtMqHjoTm6P-lvF6AqE"><meta name="google-site-verification" content="DGionF7db9g0dOgeBXwOAN2tmCzWBdo5yOdc_-5UcuE"><meta name="google-site-verification" content="Q9LlidR0toR7UtSyVO23xNeaqJmRp8I6r4ghBQTtntU"><meta name="google-site-verification" content="rQawcZaTEK_UrDG30cz_7nVKOVvBass61QEes0Tm04g"><meta name="google-site-verification" content="8L3ghjzKIj241AYAmEygniTe604tsXFkIrb1v-DBtGo"><meta name="google-site-verification" content="Osw7QcOK045GmOYJI2MM2_7AaL-s4q6pdn8gIv6JNxA"><meta name="google-site-verification" content="KiunYPvrY5x8umvAWcjhwPrB677xCar2LeT_8yaVrDg"><meta name="google-site-verification" content="b6bOMRzMVX2bJABYDGBPtpGcB_AUZ-o2SOTggQXErkg"><meta property="og:type" content="article"><meta property="og:image" content="https://colab.research.google.com/img/colab_favicon_256px.png"><meta property="og:title" content="Google Colaboratory"><meta http-equiv="origin-trial" content="AmUpB2+Hlwk73pYiEMbnkef/dprJi1I9rClec33apyFsbVOaCIRN29Rk9M4ht5Otgbp+thCc3MMD73GyCNfEWAkAAAB3eyJvcmlnaW4iOiJodHRwczovL2NvbGFiLnJlc2VhcmNoLmdvb2dsZS5jb206NDQzIiwiZmVhdHVyZSI6IkRpc2FibGVUaGlyZFBhcnR5U3RvcmFnZVBhcnRpdGlvbmluZyIsImV4cGlyeSI6MTcyNTQwNzk5OX0="><script nonce="">window.performance.mark('head_end'); window.performance.measure('head', 'head_start', 'head_end');</script><script async="" src="./untitled10_files/lazy.min.js.download" nonce=""></script><style id="inert-style">
[inert] {
  pointer-events: none;
  cursor: default;
}

[inert], [inert] * {
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
}
</style><style>
    html {
      <!---->
      <!---->
      --code-cell-background: #f7f7f7;
      --colab-anchor-color: #00e;
      --colab-border-color: #dadada;
      --colab-bold-border-color: #111;
      --colab-callout-color: var(--paper-blue-700);
      --colab-divider-color: var(--paper-grey-300);
      --colab-debugger-line-color: var(--paper-red-100);
      --colab-editor-focus-color: var(--paper-grey-200);
      --colab-highlighted-surface-color: var(--paper-grey-200);
      --colab-icon-color: var(--paper-grey-700);
      --colab-input-button-hover-background: #6161611a;
      --colab-secondary-icon-color: #93adcf;
      --colab-input-placeholder-color: var(--colab-secondary-text-color);
      --colab-primary-surface-color: var(--paper-white);
      --colab-primary-text-color: var(--paper-grey-900);
      --colab-secondary-text-color: #616161;
      --colab-tertiary-text-color: var(--paper-grey-600);
      --colab-secondary-surface-color: #f7f7f7;
      --colab-snackbar-surface-color: #333333;
      --colab-scrollbar-color: rgba(0, 0, 0, 0.1);
      --colab-gutter-icon-color: #8c8c8c;
      --colab-active-execution-icon-color: var(--paper-grey-800);
      --colab-error-icon-color: var(--paper-red-a700);
      --colab-title-color: #000;
      --colab-form-field-underline-color: var(--paper-grey-400);
      --colab-editor-focus-border-thickness: 2px;
      --colab-diff-editor-background: #fff;
      --colab-local-diff-background: #eef;
      --colab-remote-diff-background: #feffe0;
      --colab-merged-diff-background: #d7fed8;
      --colab-status-okay: var(--google-green-700);
      --ansi-black: rgb(0, 0, 0);
      --ansi-red: rgb(139, 0, 0);
      --ansi-green: rgb(0, 100, 0);
      --ansi-yellow: rgb(205, 205, 0);
      --ansi-blue: rgb(0, 0, 238);
      --ansi-magenta: rgb(205, 0, 205);
      --ansi-cyan: rgb(70, 130, 180);
      --ansi-gray: rgb(229, 229, 229);
      --ansi-bright-black: rgb(127, 127, 127);
      --ansi-bright-red: rgb(255, 0, 0);
      --ansi-bright-green: rgb(0, 208, 0);
      --ansi-bright-yellow: rgb(255, 255, 0);
      --ansi-bright-blue: rgb(92, 92, 255);
      --ansi-bright-magenta: rgb(255, 0, 255);
      --ansi-bright-cyan: rgb(0, 255, 255);
      --ansi-bright-gray: rgb(255, 255, 255);
      --gcp-success-icon-color: var(--google-green-700);
      --colab-comment-anchor-color: var(--google-blue-600);
      --colab-filled-button-ripple-color: #000;
      --colab-comment-highlight-color: #ffe082; /* material amber-200 */
      <!---->
      --colab-input-border-color: #a9a9a9;
      --colab-logo-dark: var(--google-orange-600);
      --colab-logo-light: var(--google-yellow-600);
      --colab-status-error: #e51c23;
      --colab-status-warning: #f09300;
      --colab-chrome-font-family: 'Roboto', 'Noto', sans-serif;
      --colab-google-sans-font-family: 'Google Sans', 'Roboto', 'Noto',
        sans-serif;
      --colab-chrome-font-size: 14px;
      <!---->
      --colab-icon-hover-color: var(--colab-primary-text-color);
      --colab-fresh-execution-count-color: var(--colab-primary-text-color);
      --colab-stale-execution-count-color: var(--colab-input-placeholder-color);
      --error-color: #b3261e;
      --primary-color: var(--google-blue-700);
      --google-green-700: #188038;
      --google-grey-800: #3c4043;
      --google-blue-200: #aecbfa;
      --google-blue-300: #8ab4f8;
      --google-blue-700: #1967d2;
      --google-blue-900: #174ea6;
      --hairline-button-secondary: var(--google-blue-900);
      --mdc-theme-primary: var(--primary-color);
      --mdc-theme-on-primary: var(--colab-primary-surface-color);
      --md-sys-color-primary: var(--primary-color);
      --md-sys-color-on-primary: var(--colab-primary-surface-color);
    }
    html[theme='dark'] {
      <!---->
      --colab-anchor-color: var(--paper-blue-300);
      --colab-callout-color: var(--google-blue-600);
      --code-cell-background: #1e1e1e;
      --colab-border-color: var(--paper-grey-700);
      --colab-bold-border-color: #eee;
      --colab-divider-color: var(--paper-grey-700);
      --colab-debugger-line-color: var(--paper-red-900);
      --colab-editor-focus-color: #282828;
      --colab-highlighted-surface-color: #525252;
      --colab-icon-color: var(--paper-grey-400);
      --colab-input-button-hover-background: #bdbdbd1a;
      --colab-input-placeholder-color: var(--colab-tertiary-text-color);
      --colab-primary-surface-color: #383838;
      --colab-primary-text-color: #d5d5d5;
      --colab-secondary-text-color: #f7f7f7;
      --colab-tertiary-text-color: #b3b3b3;
      --colab-secondary-surface-color: #454545;
      --colab-snackbar-surface-color: var(--colab-secondary-surface-color);
      --colab-scrollbar-color: rgba(255, 255, 255, 0.2);
      --colab-gutter-icon-color: #858585;
      --colab-active-execution-icon-color: var(--colab-icon-color);
      --colab-error-icon-color: #ef5350;
      --colab-title-color: #fff;
      --colab-form-field-underline-color: var(--paper-grey-600);
      --colab-diff-editor-background: #000;
      --colab-local-diff-background: #292935;
      --colab-remote-diff-background: #2e2f08;
      --colab-merged-diff-background: #09380b;
      --colab-status-okay: #00c752;
      --ansi-black: rgb(127, 127, 127);
      --ansi-red: rgb(255, 122, 136);
      --ansi-green: rgb(87, 187, 138);
      --ansi-yellow: rgb(255, 255, 102);
      --ansi-blue: rgb(130, 177, 255);
      --ansi-magenta: rgb(205, 0, 205);
      --ansi-cyan: rgb(153, 187, 215);
      --ansi-gray: rgb(229, 229, 229);
      --ansi-bright-green: rgb(0, 255, 0);
      --gcp-success-icon-color: #81c995;
      --colab-comment-anchor-color: var(--paper-blue-300);
      --colab-filled-button-ripple-color: #fff;
      --colab-comment-highlight-color: #1a237e; /* material indigo-900 */
      --error-color: var(--paper-deep-orange-a100);
      --primary-color: var(--google-blue-300);
      --hairline-button-secondary: var(--google-blue-200);
    }
    @media (prefers-color-scheme: dark) {
      html[theme='adaptive'] {
        <!---->
        --colab-anchor-color: var(--paper-blue-400);
        --colab-callout-color: var(--paper-blue-900);
        --code-cell-background: #1e1e1e;
        --colab-border-color: var(--paper-grey-700);
        --colab-bold-border-color: #eee;
        --colab-divider-color: var(--paper-grey-700);
        --colab-debugger-line-color: var(--paper-red-900);
        --colab-editor-focus-color: #282828;
        --colab-highlighted-surface-color: #525252;
        --colab-icon-color: var(--paper-grey-100);
        --colab-input-button-hover-background: #bdbdbd1a;
        --colab-input-placeholder-color: var(--colab-tertiary-text-color);
        --colab-primary-surface-color: #383838;
        --colab-primary-text-color: #d5d5d5;
        --colab-secondary-text-color: #f7f7f7;
        --colab-tertiary-text-color: #b3b3b3;
        --colab-secondary-surface-color: #454545;
        --colab-snackbar-surface-color: var(--colab-secondary-surface-color);
        --colab-scrollbar-color: rgba(255, 255, 255, 0.2);
        --colab-gutter-icon-color: #858585;
        --colab-execution-count: var(--colab-primary-text-color);
        --colab-active-execution-icon-color: var(--colab-icon-color);
        --colab-error-icon-color: #ef5350;
        --colab-title-color: #fff;
        --colab-form-field-underline-color: var(--paper-grey-600);
        --colab-diff-editor-background: #000;
        --colab-local-diff-background: #292935;
        --colab-remote-diff-background: #2e2f08;
        --colab-merged-diff-background: #09380b;
        --colab-status-okay: #00c752;
        --ansi-black: rgb(63, 63, 63);
        --ansi-red: rgb(255, 122, 136);
        --ansi-green: rgb(87, 187, 138);
        --ansi-yellow: rgb(255, 255, 102);
        --ansi-blue: rgb(130, 177, 255);
        --ansi-magenta: rgb(205, 0, 205);
        --ansi-cyan: rgb(153, 187, 215);
        --ansi-gray: rgb(229, 229, 229);
        --colab-filled-button-ripple-color: #fff;
        --error-color: var(--paper-deep-orange-a100);
        --primary-color: var(--google-blue-300);
        --hairline-button-secondary: var(--google-blue-200);
      }
    }
    html[editor='High Contrast Dark'] {
      --code-cell-background: #000;
      --colab-editor-focus-color: #f38518;
      --colab-editor-focus-border-thickness: 1px;
    }
    html[editor='Monokai'] {
      --code-cell-background: #272822;
    }
    html[editor='All Hallows Eve'] {
      --code-cell-background: #000;
    }
    html[editor='Amy'] {
      --code-cell-background: #200020;
    }
    html[editor='Birds Of Paradise'] {
      --code-cell-background: #372725;
    }
    html[editor='Blackboard'] {
      --code-cell-background: #0c1021;
    }
    html[editor='Clouds Midnight'] {
      --code-cell-background: #191919;
    }
    html[editor='Dominion Day'] {
      --code-cell-background: #372725;
    }
    html[editor='Espresso Libre'] {
      --code-cell-background: #2a211c;
    }
    html[editor='Merbivore'] {
      --code-cell-background: #161616;
    }
    html[editor='Night Owl'] {
      --code-cell-background: #011627;
    }
    html[editor='Oceanic Next'] {
      --code-cell-background: #1b2b34;
    }
    html[editor='Pastels On Dark'] {
      --code-cell-background: #211e1e;
    }
    html[editor='Space Cadet'] {
      --code-cell-background: #0d0d0d;
    }
    html[editor='Sunburst'] {
      --code-cell-background: #000;
    }
    html[editor='Twilight'] {
      --code-cell-background: #141414;
    }
    html[editor='Vibrant Ink'] {
      --code-cell-background: #000;
    }
    html[editor='Zenburnesque'] {
      --code-cell-background: #404040;
    }
    html[editor='Idle Fingers'] {
      --code-cell-background: #323232;
    }
    html[editor='Mono Industrial'] {
      --code-cell-background: #222c28;
    }
    html[editor='Synthwave84'] {
      --colab-primary-surface-color: #241b2f;
      --colab-secondary-surface-color: #49549539;
      --code-cell-background: #262335;
      --colab-highlighted-surface-color: #372d4b;
      --colab-border-color: #34294fb3;
      --colab-anchor-color: #f97e72;
      --ansi-red: #fe4450;
      --ansi-green: #72f1b8;
      --ansi-yellow: #f97e72;
      --ansi-blue: #03edf9;
      --ansi-magenta: #ff7edb;
      --ansi-cyan: #03edf9;
      --ansi-bright-red: #fe4450;
      --ansi-bright-green: #72f1b8;
      --ansi-bright-yellow: #fede5d;
      --ansi-bright-blue: #03edf9;
      --ansi-bright-magenta: #ff7edb;
      --ansi-bright-cyan: #03edf9;
    }
    <!---->
  <!----></style><script async="async" type="text/javascript" src="./untitled10_files/editor.main.js.download"></script><script async="" type="text/javascript" charset="UTF-8" src="./untitled10_files/rs=AA2YrTvBynad-nWEy1xIb9j1w6LpLOF6IQ" nonce=""></script><link type="text/css" rel="stylesheet" href="./untitled10_files/rs=AA2YrTssrVR1lBtzoy_MObv1DSp-vWG36A"><link rel="stylesheet" type="text/css" data-name="vs/editor/editor.main" href="./untitled10_files/editor.main.css"><script async="async" type="text/javascript" src="./untitled10_files/editor.main.nls.js.download"></script><style type="text/css">.MathJax_Hover_Frame {border-radius: .25em; -webkit-border-radius: .25em; -moz-border-radius: .25em; -khtml-border-radius: .25em; box-shadow: 0px 0px 15px #83A; -webkit-box-shadow: 0px 0px 15px #83A; -moz-box-shadow: 0px 0px 15px #83A; -khtml-box-shadow: 0px 0px 15px #83A; border: 1px solid #A6D ! important; display: inline-block; position: absolute}
.MathJax_Menu_Button .MathJax_Hover_Arrow {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 4px; -webkit-border-radius: 4px; -moz-border-radius: 4px; -khtml-border-radius: 4px; font-family: 'Courier New',Courier; font-size: 9px; color: #F0F0F0}
.MathJax_Menu_Button .MathJax_Hover_Arrow span {display: block; background-color: #AAA; border: 1px solid; border-radius: 3px; line-height: 0; padding: 4px}
.MathJax_Hover_Arrow:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_Hover_Arrow:hover span {background-color: #CCC!important}
</style><style type="text/css">#MathJax_About {position: fixed; left: 50%; width: auto; text-align: center; border: 3px outset; padding: 1em 2em; background-color: #DDDDDD; color: black; cursor: default; font-family: message-box; font-size: 120%; font-style: normal; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; border-radius: 15px; -webkit-border-radius: 15px; -moz-border-radius: 15px; -khtml-border-radius: 15px; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_About.MathJax_MousePost {outline: none}
.MathJax_Menu {position: absolute; background-color: white; color: black; width: auto; padding: 2px; border: 1px solid #CCCCCC; margin: 0; cursor: default; font: menu; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
.MathJax_MenuItem {padding: 2px 2em; background: transparent}
.MathJax_MenuArrow {position: absolute; right: .5em; padding-top: .25em; color: #666666; font-size: .75em}
.MathJax_MenuActive .MathJax_MenuArrow {color: white}
.MathJax_MenuArrow.RTL {left: .5em; right: auto}
.MathJax_MenuCheck {position: absolute; left: .7em}
.MathJax_MenuCheck.RTL {right: .7em; left: auto}
.MathJax_MenuRadioCheck {position: absolute; left: 1em}
.MathJax_MenuRadioCheck.RTL {right: 1em; left: auto}
.MathJax_MenuLabel {padding: 2px 2em 4px 1.33em; font-style: italic}
.MathJax_MenuRule {border-top: 1px solid #CCCCCC; margin: 4px 1px 0px}
.MathJax_MenuDisabled {color: GrayText}
.MathJax_MenuActive {background-color: Highlight; color: HighlightText}
.MathJax_MenuDisabled:focus, .MathJax_MenuLabel:focus {background-color: #E8E8E8}
.MathJax_ContextMenu:focus {outline: none}
.MathJax_ContextMenu .MathJax_MenuItem:focus {outline: none}
#MathJax_AboutClose {top: .2em; right: .2em}
.MathJax_Menu .MathJax_MenuClose {top: -10px; left: -10px}
.MathJax_MenuClose {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; font-family: 'Courier New',Courier; font-size: 24px; color: #F0F0F0}
.MathJax_MenuClose span {display: block; background-color: #AAA; border: 1.5px solid; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; line-height: 0; padding: 8px 0 6px}
.MathJax_MenuClose:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_MenuClose:hover span {background-color: #CCC!important}
.MathJax_MenuClose:hover:focus {outline: none}
</style><style type="text/css">.MJX_Assistive_MathML {position: absolute!important; top: 0; left: 0; clip: rect(1px, 1px, 1px, 1px); padding: 1px 0 0 0!important; border: 0!important; height: 1px!important; width: 1px!important; overflow: hidden!important; display: block!important; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none}
.MJX_Assistive_MathML.MJX_Assistive_MathML_Block {width: 100%!important}
</style><style type="text/css">#MathJax_Zoom {position: absolute; background-color: #F0F0F0; overflow: auto; display: block; z-index: 301; padding: .5em; border: 1px solid black; margin: 0; font-weight: normal; font-style: normal; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; -webkit-box-sizing: content-box; -moz-box-sizing: content-box; box-sizing: content-box; box-shadow: 5px 5px 15px #AAAAAA; -webkit-box-shadow: 5px 5px 15px #AAAAAA; -moz-box-shadow: 5px 5px 15px #AAAAAA; -khtml-box-shadow: 5px 5px 15px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_ZoomOverlay {position: absolute; left: 0; top: 0; z-index: 300; display: inline-block; width: 100%; height: 100%; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
#MathJax_ZoomFrame {position: relative; display: inline-block; height: 0; width: 0}
#MathJax_ZoomEventTrap {position: absolute; left: 0; top: 0; z-index: 302; display: inline-block; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
</style><style type="text/css">.MathJax_Preview {color: #888}
#MathJax_Message {position: fixed; left: 1em; bottom: 1.5em; background-color: #E6E6E6; border: 1px solid #959595; margin: 0px; padding: 2px 8px; z-index: 102; color: black; font-size: 80%; width: auto; white-space: nowrap}
#MathJax_MSIE_Frame {position: absolute; top: 0; left: 0; width: 0px; z-index: 101; border: 0px; margin: 0px; padding: 0px}
.MathJax_Error {color: #CC0000; font-style: italic}
</style><script async="async" type="text/javascript" src="./untitled10_files/markdown.js.download"></script><script src="./untitled10_files/api.js.download" nonce=""></script><script src="./untitled10_files/api(1).js.download" nonce=""></script><style type="text/css" media="screen" class="monaco-colors">.codicon-add:before { content: '\ea60'; }
.codicon-plus:before { content: '\ea60'; }
.codicon-gist-new:before { content: '\ea60'; }
.codicon-repo-create:before { content: '\ea60'; }
.codicon-lightbulb:before { content: '\ea61'; }
.codicon-light-bulb:before { content: '\ea61'; }
.codicon-repo:before { content: '\ea62'; }
.codicon-repo-delete:before { content: '\ea62'; }
.codicon-gist-fork:before { content: '\ea63'; }
.codicon-repo-forked:before { content: '\ea63'; }
.codicon-git-pull-request:before { content: '\ea64'; }
.codicon-git-pull-request-abandoned:before { content: '\ea64'; }
.codicon-record-keys:before { content: '\ea65'; }
.codicon-keyboard:before { content: '\ea65'; }
.codicon-tag:before { content: '\ea66'; }
.codicon-tag-add:before { content: '\ea66'; }
.codicon-tag-remove:before { content: '\ea66'; }
.codicon-person:before { content: '\ea67'; }
.codicon-person-follow:before { content: '\ea67'; }
.codicon-person-outline:before { content: '\ea67'; }
.codicon-person-filled:before { content: '\ea67'; }
.codicon-git-branch:before { content: '\ea68'; }
.codicon-git-branch-create:before { content: '\ea68'; }
.codicon-git-branch-delete:before { content: '\ea68'; }
.codicon-source-control:before { content: '\ea68'; }
.codicon-mirror:before { content: '\ea69'; }
.codicon-mirror-public:before { content: '\ea69'; }
.codicon-star:before { content: '\ea6a'; }
.codicon-star-add:before { content: '\ea6a'; }
.codicon-star-delete:before { content: '\ea6a'; }
.codicon-star-empty:before { content: '\ea6a'; }
.codicon-comment:before { content: '\ea6b'; }
.codicon-comment-add:before { content: '\ea6b'; }
.codicon-alert:before { content: '\ea6c'; }
.codicon-warning:before { content: '\ea6c'; }
.codicon-search:before { content: '\ea6d'; }
.codicon-search-save:before { content: '\ea6d'; }
.codicon-log-out:before { content: '\ea6e'; }
.codicon-sign-out:before { content: '\ea6e'; }
.codicon-log-in:before { content: '\ea6f'; }
.codicon-sign-in:before { content: '\ea6f'; }
.codicon-eye:before { content: '\ea70'; }
.codicon-eye-unwatch:before { content: '\ea70'; }
.codicon-eye-watch:before { content: '\ea70'; }
.codicon-circle-filled:before { content: '\ea71'; }
.codicon-primitive-dot:before { content: '\ea71'; }
.codicon-close-dirty:before { content: '\ea71'; }
.codicon-debug-breakpoint:before { content: '\ea71'; }
.codicon-debug-breakpoint-disabled:before { content: '\ea71'; }
.codicon-debug-hint:before { content: '\ea71'; }
.codicon-primitive-square:before { content: '\ea72'; }
.codicon-edit:before { content: '\ea73'; }
.codicon-pencil:before { content: '\ea73'; }
.codicon-info:before { content: '\ea74'; }
.codicon-issue-opened:before { content: '\ea74'; }
.codicon-gist-private:before { content: '\ea75'; }
.codicon-git-fork-private:before { content: '\ea75'; }
.codicon-lock:before { content: '\ea75'; }
.codicon-mirror-private:before { content: '\ea75'; }
.codicon-close:before { content: '\ea76'; }
.codicon-remove-close:before { content: '\ea76'; }
.codicon-x:before { content: '\ea76'; }
.codicon-repo-sync:before { content: '\ea77'; }
.codicon-sync:before { content: '\ea77'; }
.codicon-clone:before { content: '\ea78'; }
.codicon-desktop-download:before { content: '\ea78'; }
.codicon-beaker:before { content: '\ea79'; }
.codicon-microscope:before { content: '\ea79'; }
.codicon-vm:before { content: '\ea7a'; }
.codicon-device-desktop:before { content: '\ea7a'; }
.codicon-file:before { content: '\ea7b'; }
.codicon-file-text:before { content: '\ea7b'; }
.codicon-more:before { content: '\ea7c'; }
.codicon-ellipsis:before { content: '\ea7c'; }
.codicon-kebab-horizontal:before { content: '\ea7c'; }
.codicon-mail-reply:before { content: '\ea7d'; }
.codicon-reply:before { content: '\ea7d'; }
.codicon-organization:before { content: '\ea7e'; }
.codicon-organization-filled:before { content: '\ea7e'; }
.codicon-organization-outline:before { content: '\ea7e'; }
.codicon-new-file:before { content: '\ea7f'; }
.codicon-file-add:before { content: '\ea7f'; }
.codicon-new-folder:before { content: '\ea80'; }
.codicon-file-directory-create:before { content: '\ea80'; }
.codicon-trash:before { content: '\ea81'; }
.codicon-trashcan:before { content: '\ea81'; }
.codicon-history:before { content: '\ea82'; }
.codicon-clock:before { content: '\ea82'; }
.codicon-folder:before { content: '\ea83'; }
.codicon-file-directory:before { content: '\ea83'; }
.codicon-symbol-folder:before { content: '\ea83'; }
.codicon-logo-github:before { content: '\ea84'; }
.codicon-mark-github:before { content: '\ea84'; }
.codicon-github:before { content: '\ea84'; }
.codicon-terminal:before { content: '\ea85'; }
.codicon-console:before { content: '\ea85'; }
.codicon-repl:before { content: '\ea85'; }
.codicon-zap:before { content: '\ea86'; }
.codicon-symbol-event:before { content: '\ea86'; }
.codicon-error:before { content: '\ea87'; }
.codicon-stop:before { content: '\ea87'; }
.codicon-variable:before { content: '\ea88'; }
.codicon-symbol-variable:before { content: '\ea88'; }
.codicon-array:before { content: '\ea8a'; }
.codicon-symbol-array:before { content: '\ea8a'; }
.codicon-symbol-module:before { content: '\ea8b'; }
.codicon-symbol-package:before { content: '\ea8b'; }
.codicon-symbol-namespace:before { content: '\ea8b'; }
.codicon-symbol-object:before { content: '\ea8b'; }
.codicon-symbol-method:before { content: '\ea8c'; }
.codicon-symbol-function:before { content: '\ea8c'; }
.codicon-symbol-constructor:before { content: '\ea8c'; }
.codicon-symbol-boolean:before { content: '\ea8f'; }
.codicon-symbol-null:before { content: '\ea8f'; }
.codicon-symbol-numeric:before { content: '\ea90'; }
.codicon-symbol-number:before { content: '\ea90'; }
.codicon-symbol-structure:before { content: '\ea91'; }
.codicon-symbol-struct:before { content: '\ea91'; }
.codicon-symbol-parameter:before { content: '\ea92'; }
.codicon-symbol-type-parameter:before { content: '\ea92'; }
.codicon-symbol-key:before { content: '\ea93'; }
.codicon-symbol-text:before { content: '\ea93'; }
.codicon-symbol-reference:before { content: '\ea94'; }
.codicon-go-to-file:before { content: '\ea94'; }
.codicon-symbol-enum:before { content: '\ea95'; }
.codicon-symbol-value:before { content: '\ea95'; }
.codicon-symbol-ruler:before { content: '\ea96'; }
.codicon-symbol-unit:before { content: '\ea96'; }
.codicon-activate-breakpoints:before { content: '\ea97'; }
.codicon-archive:before { content: '\ea98'; }
.codicon-arrow-both:before { content: '\ea99'; }
.codicon-arrow-down:before { content: '\ea9a'; }
.codicon-arrow-left:before { content: '\ea9b'; }
.codicon-arrow-right:before { content: '\ea9c'; }
.codicon-arrow-small-down:before { content: '\ea9d'; }
.codicon-arrow-small-left:before { content: '\ea9e'; }
.codicon-arrow-small-right:before { content: '\ea9f'; }
.codicon-arrow-small-up:before { content: '\eaa0'; }
.codicon-arrow-up:before { content: '\eaa1'; }
.codicon-bell:before { content: '\eaa2'; }
.codicon-bold:before { content: '\eaa3'; }
.codicon-book:before { content: '\eaa4'; }
.codicon-bookmark:before { content: '\eaa5'; }
.codicon-debug-breakpoint-conditional-unverified:before { content: '\eaa6'; }
.codicon-debug-breakpoint-conditional:before { content: '\eaa7'; }
.codicon-debug-breakpoint-conditional-disabled:before { content: '\eaa7'; }
.codicon-debug-breakpoint-data-unverified:before { content: '\eaa8'; }
.codicon-debug-breakpoint-data:before { content: '\eaa9'; }
.codicon-debug-breakpoint-data-disabled:before { content: '\eaa9'; }
.codicon-debug-breakpoint-log-unverified:before { content: '\eaaa'; }
.codicon-debug-breakpoint-log:before { content: '\eaab'; }
.codicon-debug-breakpoint-log-disabled:before { content: '\eaab'; }
.codicon-briefcase:before { content: '\eaac'; }
.codicon-broadcast:before { content: '\eaad'; }
.codicon-browser:before { content: '\eaae'; }
.codicon-bug:before { content: '\eaaf'; }
.codicon-calendar:before { content: '\eab0'; }
.codicon-case-sensitive:before { content: '\eab1'; }
.codicon-check:before { content: '\eab2'; }
.codicon-checklist:before { content: '\eab3'; }
.codicon-chevron-down:before { content: '\eab4'; }
.codicon-drop-down-button:before { content: '\eab4'; }
.codicon-chevron-left:before { content: '\eab5'; }
.codicon-chevron-right:before { content: '\eab6'; }
.codicon-chevron-up:before { content: '\eab7'; }
.codicon-chrome-close:before { content: '\eab8'; }
.codicon-chrome-maximize:before { content: '\eab9'; }
.codicon-chrome-minimize:before { content: '\eaba'; }
.codicon-chrome-restore:before { content: '\eabb'; }
.codicon-circle:before { content: '\eabc'; }
.codicon-circle-outline:before { content: '\eabc'; }
.codicon-debug-breakpoint-unverified:before { content: '\eabc'; }
.codicon-circle-slash:before { content: '\eabd'; }
.codicon-circuit-board:before { content: '\eabe'; }
.codicon-clear-all:before { content: '\eabf'; }
.codicon-clippy:before { content: '\eac0'; }
.codicon-close-all:before { content: '\eac1'; }
.codicon-cloud-download:before { content: '\eac2'; }
.codicon-cloud-upload:before { content: '\eac3'; }
.codicon-code:before { content: '\eac4'; }
.codicon-collapse-all:before { content: '\eac5'; }
.codicon-color-mode:before { content: '\eac6'; }
.codicon-comment-discussion:before { content: '\eac7'; }
.codicon-compare-changes:before { content: '\eafd'; }
.codicon-credit-card:before { content: '\eac9'; }
.codicon-dash:before { content: '\eacc'; }
.codicon-dashboard:before { content: '\eacd'; }
.codicon-database:before { content: '\eace'; }
.codicon-debug-continue:before { content: '\eacf'; }
.codicon-debug-disconnect:before { content: '\ead0'; }
.codicon-debug-pause:before { content: '\ead1'; }
.codicon-debug-restart:before { content: '\ead2'; }
.codicon-debug-start:before { content: '\ead3'; }
.codicon-debug-step-into:before { content: '\ead4'; }
.codicon-debug-step-out:before { content: '\ead5'; }
.codicon-debug-step-over:before { content: '\ead6'; }
.codicon-debug-stop:before { content: '\ead7'; }
.codicon-debug:before { content: '\ead8'; }
.codicon-device-camera-video:before { content: '\ead9'; }
.codicon-device-camera:before { content: '\eada'; }
.codicon-device-mobile:before { content: '\eadb'; }
.codicon-diff-added:before { content: '\eadc'; }
.codicon-diff-ignored:before { content: '\eadd'; }
.codicon-diff-modified:before { content: '\eade'; }
.codicon-diff-removed:before { content: '\eadf'; }
.codicon-diff-renamed:before { content: '\eae0'; }
.codicon-diff:before { content: '\eae1'; }
.codicon-discard:before { content: '\eae2'; }
.codicon-editor-layout:before { content: '\eae3'; }
.codicon-empty-window:before { content: '\eae4'; }
.codicon-exclude:before { content: '\eae5'; }
.codicon-extensions:before { content: '\eae6'; }
.codicon-eye-closed:before { content: '\eae7'; }
.codicon-file-binary:before { content: '\eae8'; }
.codicon-file-code:before { content: '\eae9'; }
.codicon-file-media:before { content: '\eaea'; }
.codicon-file-pdf:before { content: '\eaeb'; }
.codicon-file-submodule:before { content: '\eaec'; }
.codicon-file-symlink-directory:before { content: '\eaed'; }
.codicon-file-symlink-file:before { content: '\eaee'; }
.codicon-file-zip:before { content: '\eaef'; }
.codicon-files:before { content: '\eaf0'; }
.codicon-filter:before { content: '\eaf1'; }
.codicon-flame:before { content: '\eaf2'; }
.codicon-fold-down:before { content: '\eaf3'; }
.codicon-fold-up:before { content: '\eaf4'; }
.codicon-fold:before { content: '\eaf5'; }
.codicon-folder-active:before { content: '\eaf6'; }
.codicon-folder-opened:before { content: '\eaf7'; }
.codicon-gear:before { content: '\eaf8'; }
.codicon-gift:before { content: '\eaf9'; }
.codicon-gist-secret:before { content: '\eafa'; }
.codicon-gist:before { content: '\eafb'; }
.codicon-git-commit:before { content: '\eafc'; }
.codicon-git-compare:before { content: '\eafd'; }
.codicon-git-merge:before { content: '\eafe'; }
.codicon-github-action:before { content: '\eaff'; }
.codicon-github-alt:before { content: '\eb00'; }
.codicon-globe:before { content: '\eb01'; }
.codicon-grabber:before { content: '\eb02'; }
.codicon-graph:before { content: '\eb03'; }
.codicon-gripper:before { content: '\eb04'; }
.codicon-heart:before { content: '\eb05'; }
.codicon-home:before { content: '\eb06'; }
.codicon-horizontal-rule:before { content: '\eb07'; }
.codicon-hubot:before { content: '\eb08'; }
.codicon-inbox:before { content: '\eb09'; }
.codicon-issue-closed:before { content: '\eba4'; }
.codicon-issue-reopened:before { content: '\eb0b'; }
.codicon-issues:before { content: '\eb0c'; }
.codicon-italic:before { content: '\eb0d'; }
.codicon-jersey:before { content: '\eb0e'; }
.codicon-json:before { content: '\eb0f'; }
.codicon-bracket:before { content: '\eb0f'; }
.codicon-kebab-vertical:before { content: '\eb10'; }
.codicon-key:before { content: '\eb11'; }
.codicon-law:before { content: '\eb12'; }
.codicon-lightbulb-autofix:before { content: '\eb13'; }
.codicon-link-external:before { content: '\eb14'; }
.codicon-link:before { content: '\eb15'; }
.codicon-list-ordered:before { content: '\eb16'; }
.codicon-list-unordered:before { content: '\eb17'; }
.codicon-live-share:before { content: '\eb18'; }
.codicon-loading:before { content: '\eb19'; }
.codicon-location:before { content: '\eb1a'; }
.codicon-mail-read:before { content: '\eb1b'; }
.codicon-mail:before { content: '\eb1c'; }
.codicon-markdown:before { content: '\eb1d'; }
.codicon-megaphone:before { content: '\eb1e'; }
.codicon-mention:before { content: '\eb1f'; }
.codicon-milestone:before { content: '\eb20'; }
.codicon-mortar-board:before { content: '\eb21'; }
.codicon-move:before { content: '\eb22'; }
.codicon-multiple-windows:before { content: '\eb23'; }
.codicon-mute:before { content: '\eb24'; }
.codicon-no-newline:before { content: '\eb25'; }
.codicon-note:before { content: '\eb26'; }
.codicon-octoface:before { content: '\eb27'; }
.codicon-open-preview:before { content: '\eb28'; }
.codicon-package:before { content: '\eb29'; }
.codicon-paintcan:before { content: '\eb2a'; }
.codicon-pin:before { content: '\eb2b'; }
.codicon-play:before { content: '\eb2c'; }
.codicon-run:before { content: '\eb2c'; }
.codicon-plug:before { content: '\eb2d'; }
.codicon-preserve-case:before { content: '\eb2e'; }
.codicon-preview:before { content: '\eb2f'; }
.codicon-project:before { content: '\eb30'; }
.codicon-pulse:before { content: '\eb31'; }
.codicon-question:before { content: '\eb32'; }
.codicon-quote:before { content: '\eb33'; }
.codicon-radio-tower:before { content: '\eb34'; }
.codicon-reactions:before { content: '\eb35'; }
.codicon-references:before { content: '\eb36'; }
.codicon-refresh:before { content: '\eb37'; }
.codicon-regex:before { content: '\eb38'; }
.codicon-remote-explorer:before { content: '\eb39'; }
.codicon-remote:before { content: '\eb3a'; }
.codicon-remove:before { content: '\eb3b'; }
.codicon-replace-all:before { content: '\eb3c'; }
.codicon-replace:before { content: '\eb3d'; }
.codicon-repo-clone:before { content: '\eb3e'; }
.codicon-repo-force-push:before { content: '\eb3f'; }
.codicon-repo-pull:before { content: '\eb40'; }
.codicon-repo-push:before { content: '\eb41'; }
.codicon-report:before { content: '\eb42'; }
.codicon-request-changes:before { content: '\eb43'; }
.codicon-rocket:before { content: '\eb44'; }
.codicon-root-folder-opened:before { content: '\eb45'; }
.codicon-root-folder:before { content: '\eb46'; }
.codicon-rss:before { content: '\eb47'; }
.codicon-ruby:before { content: '\eb48'; }
.codicon-save-all:before { content: '\eb49'; }
.codicon-save-as:before { content: '\eb4a'; }
.codicon-save:before { content: '\eb4b'; }
.codicon-screen-full:before { content: '\eb4c'; }
.codicon-screen-normal:before { content: '\eb4d'; }
.codicon-search-stop:before { content: '\eb4e'; }
.codicon-server:before { content: '\eb50'; }
.codicon-settings-gear:before { content: '\eb51'; }
.codicon-settings:before { content: '\eb52'; }
.codicon-shield:before { content: '\eb53'; }
.codicon-smiley:before { content: '\eb54'; }
.codicon-sort-precedence:before { content: '\eb55'; }
.codicon-split-horizontal:before { content: '\eb56'; }
.codicon-split-vertical:before { content: '\eb57'; }
.codicon-squirrel:before { content: '\eb58'; }
.codicon-star-full:before { content: '\eb59'; }
.codicon-star-half:before { content: '\eb5a'; }
.codicon-symbol-class:before { content: '\eb5b'; }
.codicon-symbol-color:before { content: '\eb5c'; }
.codicon-symbol-customcolor:before { content: '\eb5c'; }
.codicon-symbol-constant:before { content: '\eb5d'; }
.codicon-symbol-enum-member:before { content: '\eb5e'; }
.codicon-symbol-field:before { content: '\eb5f'; }
.codicon-symbol-file:before { content: '\eb60'; }
.codicon-symbol-interface:before { content: '\eb61'; }
.codicon-symbol-keyword:before { content: '\eb62'; }
.codicon-symbol-misc:before { content: '\eb63'; }
.codicon-symbol-operator:before { content: '\eb64'; }
.codicon-symbol-property:before { content: '\eb65'; }
.codicon-wrench:before { content: '\eb65'; }
.codicon-wrench-subaction:before { content: '\eb65'; }
.codicon-symbol-snippet:before { content: '\eb66'; }
.codicon-tasklist:before { content: '\eb67'; }
.codicon-telescope:before { content: '\eb68'; }
.codicon-text-size:before { content: '\eb69'; }
.codicon-three-bars:before { content: '\eb6a'; }
.codicon-thumbsdown:before { content: '\eb6b'; }
.codicon-thumbsup:before { content: '\eb6c'; }
.codicon-tools:before { content: '\eb6d'; }
.codicon-triangle-down:before { content: '\eb6e'; }
.codicon-triangle-left:before { content: '\eb6f'; }
.codicon-triangle-right:before { content: '\eb70'; }
.codicon-triangle-up:before { content: '\eb71'; }
.codicon-twitter:before { content: '\eb72'; }
.codicon-unfold:before { content: '\eb73'; }
.codicon-unlock:before { content: '\eb74'; }
.codicon-unmute:before { content: '\eb75'; }
.codicon-unverified:before { content: '\eb76'; }
.codicon-verified:before { content: '\eb77'; }
.codicon-versions:before { content: '\eb78'; }
.codicon-vm-active:before { content: '\eb79'; }
.codicon-vm-outline:before { content: '\eb7a'; }
.codicon-vm-running:before { content: '\eb7b'; }
.codicon-watch:before { content: '\eb7c'; }
.codicon-whitespace:before { content: '\eb7d'; }
.codicon-whole-word:before { content: '\eb7e'; }
.codicon-window:before { content: '\eb7f'; }
.codicon-word-wrap:before { content: '\eb80'; }
.codicon-zoom-in:before { content: '\eb81'; }
.codicon-zoom-out:before { content: '\eb82'; }
.codicon-list-filter:before { content: '\eb83'; }
.codicon-list-flat:before { content: '\eb84'; }
.codicon-list-selection:before { content: '\eb85'; }
.codicon-selection:before { content: '\eb85'; }
.codicon-list-tree:before { content: '\eb86'; }
.codicon-debug-breakpoint-function-unverified:before { content: '\eb87'; }
.codicon-debug-breakpoint-function:before { content: '\eb88'; }
.codicon-debug-breakpoint-function-disabled:before { content: '\eb88'; }
.codicon-debug-stackframe-active:before { content: '\eb89'; }
.codicon-circle-small-filled:before { content: '\eb8a'; }
.codicon-debug-stackframe-dot:before { content: '\eb8a'; }
.codicon-debug-stackframe:before { content: '\eb8b'; }
.codicon-debug-stackframe-focused:before { content: '\eb8b'; }
.codicon-debug-breakpoint-unsupported:before { content: '\eb8c'; }
.codicon-symbol-string:before { content: '\eb8d'; }
.codicon-debug-reverse-continue:before { content: '\eb8e'; }
.codicon-debug-step-back:before { content: '\eb8f'; }
.codicon-debug-restart-frame:before { content: '\eb90'; }
.codicon-call-incoming:before { content: '\eb92'; }
.codicon-call-outgoing:before { content: '\eb93'; }
.codicon-menu:before { content: '\eb94'; }
.codicon-expand-all:before { content: '\eb95'; }
.codicon-feedback:before { content: '\eb96'; }
.codicon-group-by-ref-type:before { content: '\eb97'; }
.codicon-ungroup-by-ref-type:before { content: '\eb98'; }
.codicon-account:before { content: '\eb99'; }
.codicon-bell-dot:before { content: '\eb9a'; }
.codicon-debug-console:before { content: '\eb9b'; }
.codicon-library:before { content: '\eb9c'; }
.codicon-output:before { content: '\eb9d'; }
.codicon-run-all:before { content: '\eb9e'; }
.codicon-sync-ignored:before { content: '\eb9f'; }
.codicon-pinned:before { content: '\eba0'; }
.codicon-github-inverted:before { content: '\eba1'; }
.codicon-debug-alt:before { content: '\eb91'; }
.codicon-server-process:before { content: '\eba2'; }
.codicon-server-environment:before { content: '\eba3'; }
.codicon-pass:before { content: '\eba4'; }
.codicon-stop-circle:before { content: '\eba5'; }
.codicon-play-circle:before { content: '\eba6'; }
.codicon-record:before { content: '\eba7'; }
.codicon-debug-alt-small:before { content: '\eba8'; }
.codicon-vm-connect:before { content: '\eba9'; }
.codicon-cloud:before { content: '\ebaa'; }
.codicon-merge:before { content: '\ebab'; }
.codicon-export:before { content: '\ebac'; }
.codicon-graph-left:before { content: '\ebad'; }
.codicon-magnet:before { content: '\ebae'; }
.codicon-notebook:before { content: '\ebaf'; }
.codicon-redo:before { content: '\ebb0'; }
.codicon-check-all:before { content: '\ebb1'; }
.codicon-pinned-dirty:before { content: '\ebb2'; }
.codicon-pass-filled:before { content: '\ebb3'; }
.codicon-circle-large-filled:before { content: '\ebb4'; }
.codicon-circle-large:before { content: '\ebb5'; }
.codicon-circle-large-outline:before { content: '\ebb5'; }
.codicon-combine:before { content: '\ebb6'; }
.codicon-gather:before { content: '\ebb6'; }
.codicon-table:before { content: '\ebb7'; }
.codicon-variable-group:before { content: '\ebb8'; }
.codicon-type-hierarchy:before { content: '\ebb9'; }
.codicon-type-hierarchy-sub:before { content: '\ebba'; }
.codicon-type-hierarchy-super:before { content: '\ebbb'; }
.codicon-git-pull-request-create:before { content: '\ebbc'; }
.codicon-run-above:before { content: '\ebbd'; }
.codicon-run-below:before { content: '\ebbe'; }
.codicon-notebook-template:before { content: '\ebbf'; }
.codicon-debug-rerun:before { content: '\ebc0'; }
.codicon-workspace-trusted:before { content: '\ebc1'; }
.codicon-workspace-untrusted:before { content: '\ebc2'; }
.codicon-workspace-unspecified:before { content: '\ebc3'; }
.codicon-terminal-cmd:before { content: '\ebc4'; }
.codicon-terminal-debian:before { content: '\ebc5'; }
.codicon-terminal-linux:before { content: '\ebc6'; }
.codicon-terminal-powershell:before { content: '\ebc7'; }
.codicon-terminal-tmux:before { content: '\ebc8'; }
.codicon-terminal-ubuntu:before { content: '\ebc9'; }
.codicon-terminal-bash:before { content: '\ebca'; }
.codicon-arrow-swap:before { content: '\ebcb'; }
.codicon-copy:before { content: '\ebcc'; }
.codicon-person-add:before { content: '\ebcd'; }
.codicon-filter-filled:before { content: '\ebce'; }
.codicon-wand:before { content: '\ebcf'; }
.codicon-debug-line-by-line:before { content: '\ebd0'; }
.codicon-inspect:before { content: '\ebd1'; }
.codicon-layers:before { content: '\ebd2'; }
.codicon-layers-dot:before { content: '\ebd3'; }
.codicon-layers-active:before { content: '\ebd4'; }
.codicon-compass:before { content: '\ebd5'; }
.codicon-compass-dot:before { content: '\ebd6'; }
.codicon-compass-active:before { content: '\ebd7'; }
.codicon-azure:before { content: '\ebd8'; }
.codicon-issue-draft:before { content: '\ebd9'; }
.codicon-git-pull-request-closed:before { content: '\ebda'; }
.codicon-git-pull-request-draft:before { content: '\ebdb'; }
.codicon-debug-all:before { content: '\ebdc'; }
.codicon-debug-coverage:before { content: '\ebdd'; }
.codicon-run-errors:before { content: '\ebde'; }
.codicon-folder-library:before { content: '\ebdf'; }
.codicon-debug-continue-small:before { content: '\ebe0'; }
.codicon-beaker-stop:before { content: '\ebe1'; }
.codicon-graph-line:before { content: '\ebe2'; }
.codicon-graph-scatter:before { content: '\ebe3'; }
.codicon-pie-chart:before { content: '\ebe4'; }
.codicon-bracket-dot:before { content: '\ebe5'; }
.codicon-bracket-error:before { content: '\ebe6'; }
.codicon-lock-small:before { content: '\ebe7'; }
.codicon-azure-devops:before { content: '\ebe8'; }
.codicon-verified-filled:before { content: '\ebe9'; }
.codicon-newline:before { content: '\ebea'; }
.codicon-layout:before { content: '\ebeb'; }
.codicon-layout-activitybar-left:before { content: '\ebec'; }
.codicon-layout-activitybar-right:before { content: '\ebed'; }
.codicon-layout-panel-left:before { content: '\ebee'; }
.codicon-layout-panel-center:before { content: '\ebef'; }
.codicon-layout-panel-justify:before { content: '\ebf0'; }
.codicon-layout-panel-right:before { content: '\ebf1'; }
.codicon-layout-panel:before { content: '\ebf2'; }
.codicon-layout-sidebar-left:before { content: '\ebf3'; }
.codicon-layout-sidebar-right:before { content: '\ebf4'; }
.codicon-layout-statusbar:before { content: '\ebf5'; }
.codicon-layout-menubar:before { content: '\ebf6'; }
.codicon-layout-centered:before { content: '\ebf7'; }
.codicon-layout-sidebar-right-off:before { content: '\ec00'; }
.codicon-layout-panel-off:before { content: '\ec01'; }
.codicon-layout-sidebar-left-off:before { content: '\ec02'; }
.codicon-target:before { content: '\ebf8'; }
.codicon-indent:before { content: '\ebf9'; }
.codicon-record-small:before { content: '\ebfa'; }
.codicon-error-small:before { content: '\ebfb'; }
.codicon-arrow-circle-down:before { content: '\ebfc'; }
.codicon-arrow-circle-left:before { content: '\ebfd'; }
.codicon-arrow-circle-right:before { content: '\ebfe'; }
.codicon-arrow-circle-up:before { content: '\ebff'; }
.codicon-heart-filled:before { content: '\ec04'; }
.codicon-map:before { content: '\ec05'; }
.codicon-map-filled:before { content: '\ec06'; }
.codicon-circle-small:before { content: '\ec07'; }
.codicon-bell-slash:before { content: '\ec08'; }
.codicon-bell-slash-dot:before { content: '\ec09'; }
.codicon-comment-unresolved:before { content: '\ec0a'; }
.codicon-git-pull-request-go-to-changes:before { content: '\ec0b'; }
.codicon-git-pull-request-new-changes:before { content: '\ec0c'; }
.codicon-search-fuzzy:before { content: '\ec0d'; }
.codicon-comment-draft:before { content: '\ec0e'; }
.codicon-send:before { content: '\ec0f'; }
.codicon-sparkle:before { content: '\ec10'; }
.codicon-insert:before { content: '\ec11'; }
.codicon-dialog-error:before { content: '\ea87'; }
.codicon-dialog-warning:before { content: '\ea6c'; }
.codicon-dialog-info:before { content: '\ea74'; }
.codicon-dialog-close:before { content: '\ea76'; }
.codicon-tree-item-expanded:before { content: '\eab4'; }
.codicon-tree-filter-on-type-on:before { content: '\eb83'; }
.codicon-tree-filter-on-type-off:before { content: '\eb85'; }
.codicon-tree-filter-clear:before { content: '\ea76'; }
.codicon-tree-item-loading:before { content: '\eb19'; }
.codicon-menu-selection:before { content: '\eab2'; }
.codicon-menu-submenu:before { content: '\eab6'; }
.codicon-menubar-more:before { content: '\ea7c'; }
.codicon-scrollbar-button-left:before { content: '\eb6f'; }
.codicon-scrollbar-button-right:before { content: '\eb70'; }
.codicon-scrollbar-button-up:before { content: '\eb71'; }
.codicon-scrollbar-button-down:before { content: '\eb6e'; }
.codicon-toolbar-more:before { content: '\ea7c'; }
.codicon-quick-input-back:before { content: '\ea9b'; }
.codicon-widget-close:before { content: '\ea76'; }
.codicon-goto-previous-location:before { content: '\eaa1'; }
.codicon-goto-next-location:before { content: '\ea9a'; }
.codicon-diff-review-insert:before { content: '\ea60'; }
.codicon-diff-review-remove:before { content: '\eb3b'; }
.codicon-diff-review-close:before { content: '\ea76'; }
.codicon-parameter-hints-next:before { content: '\eab4'; }
.codicon-parameter-hints-previous:before { content: '\eab7'; }
.codicon-suggest-more-info:before { content: '\eab6'; }
.codicon-inline-suggestion-hints-next:before { content: '\eab6'; }
.codicon-inline-suggestion-hints-previous:before { content: '\eab5'; }
.codicon-diff-insert:before { content: '\ea60'; }
.codicon-diff-remove:before { content: '\eb3b'; }
.codicon-find-selection:before { content: '\eb85'; }
.codicon-find-collapsed:before { content: '\eab6'; }
.codicon-find-expanded:before { content: '\eab4'; }
.codicon-find-replace:before { content: '\eb3d'; }
.codicon-find-replace-all:before { content: '\eb3c'; }
.codicon-find-previous-match:before { content: '\eaa1'; }
.codicon-find-next-match:before { content: '\ea9a'; }
.codicon-folding-expanded:before { content: '\eab4'; }
.codicon-folding-collapsed:before { content: '\eab6'; }
.codicon-folding-manual-collapsed:before { content: '\eab6'; }
.codicon-folding-manual-expanded:before { content: '\eab4'; }
.codicon-marker-navigation-next:before { content: '\ea9a'; }
.codicon-marker-navigation-previous:before { content: '\eaa1'; }
.codicon-extensions-warning-message:before { content: '\ea6c'; }
.monaco-editor .inputarea.ime-input { background-color: #1e1e1e; }
.monaco-editor .view-overlays .current-line { border: 2px solid #282828; }
.monaco-editor .margin-view-overlays .current-line-margin { border: 2px solid #282828; }
.monaco-editor .bracket-indent-guide.lvl-0 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-1 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-2 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-3 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-4 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-5 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-6 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-7 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-8 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-9 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-10 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-11 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-12 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-13 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-14 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-15 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-16 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-17 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-18 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-19 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-20 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-21 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-22 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-23 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-24 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-25 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-26 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .bracket-indent-guide.lvl-27 { --guide-color: rgba(255, 215, 0, 0.3); --guide-color-active: #ffd700; }
.monaco-editor .bracket-indent-guide.lvl-28 { --guide-color: rgba(218, 112, 214, 0.3); --guide-color-active: #da70d6; }
.monaco-editor .bracket-indent-guide.lvl-29 { --guide-color: rgba(23, 159, 255, 0.3); --guide-color-active: #179fff; }
.monaco-editor .vertical { box-shadow: 1px 0 0 0 var(--guide-color) inset; }
.monaco-editor .horizontal-top { border-top: 1px solid var(--guide-color); }
.monaco-editor .horizontal-bottom { border-bottom: 1px solid var(--guide-color); }
.monaco-editor .vertical.indent-active { box-shadow: 1px 0 0 0 var(--guide-color-active) inset; }
.monaco-editor .horizontal-top.indent-active { border-top: 1px solid var(--guide-color-active); }
.monaco-editor .horizontal-bottom.indent-active { border-bottom: 1px solid var(--guide-color-active); }
.monaco-editor .line-numbers.dimmed-line-number { color: rgba(133, 133, 133, 0.4); }
.monaco-editor .cursors-layer .cursor { background-color: #aeafad; border-color: #aeafad; color: #515052; }
.monaco-editor .unexpected-closing-bracket { color: rgba(255, 18, 18, 0.8); }
.monaco-editor .bracket-highlighting-0 { color: #ffd700; }
.monaco-editor .bracket-highlighting-1 { color: #da70d6; }
.monaco-editor .bracket-highlighting-2 { color: #179fff; }
.monaco-editor .bracket-highlighting-3 { color: #ffd700; }
.monaco-editor .bracket-highlighting-4 { color: #da70d6; }
.monaco-editor .bracket-highlighting-5 { color: #179fff; }
.monaco-editor .bracket-highlighting-6 { color: #ffd700; }
.monaco-editor .bracket-highlighting-7 { color: #da70d6; }
.monaco-editor .bracket-highlighting-8 { color: #179fff; }
.monaco-editor .bracket-highlighting-9 { color: #ffd700; }
.monaco-editor .bracket-highlighting-10 { color: #da70d6; }
.monaco-editor .bracket-highlighting-11 { color: #179fff; }
.monaco-editor .bracket-highlighting-12 { color: #ffd700; }
.monaco-editor .bracket-highlighting-13 { color: #da70d6; }
.monaco-editor .bracket-highlighting-14 { color: #179fff; }
.monaco-editor .bracket-highlighting-15 { color: #ffd700; }
.monaco-editor .bracket-highlighting-16 { color: #da70d6; }
.monaco-editor .bracket-highlighting-17 { color: #179fff; }
.monaco-editor .bracket-highlighting-18 { color: #ffd700; }
.monaco-editor .bracket-highlighting-19 { color: #da70d6; }
.monaco-editor .bracket-highlighting-20 { color: #179fff; }
.monaco-editor .bracket-highlighting-21 { color: #ffd700; }
.monaco-editor .bracket-highlighting-22 { color: #da70d6; }
.monaco-editor .bracket-highlighting-23 { color: #179fff; }
.monaco-editor .bracket-highlighting-24 { color: #ffd700; }
.monaco-editor .bracket-highlighting-25 { color: #da70d6; }
.monaco-editor .bracket-highlighting-26 { color: #179fff; }
.monaco-editor .bracket-highlighting-27 { color: #ffd700; }
.monaco-editor .bracket-highlighting-28 { color: #da70d6; }
.monaco-editor .bracket-highlighting-29 { color: #179fff; }
.monaco-editor .squiggly-error { background: url("data:image/svg+xml,%3Csvg%20xmlns%3D'http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg'%20viewBox%3D'0%200%206%203'%20enable-background%3D'new%200%200%206%203'%20height%3D'3'%20width%3D'6'%3E%3Cg%20fill%3D'%23f14c4c'%3E%3Cpolygon%20points%3D'5.5%2C0%202.5%2C3%201.1%2C3%204.1%2C0'%2F%3E%3Cpolygon%20points%3D'4%2C0%206%2C2%206%2C0.6%205.4%2C0'%2F%3E%3Cpolygon%20points%3D'0%2C2%201%2C3%202.4%2C3%200%2C0.6'%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E") repeat-x bottom left; }
.monaco-editor .squiggly-warning { background: url("data:image/svg+xml,%3Csvg%20xmlns%3D'http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg'%20viewBox%3D'0%200%206%203'%20enable-background%3D'new%200%200%206%203'%20height%3D'3'%20width%3D'6'%3E%3Cg%20fill%3D'%23cca700'%3E%3Cpolygon%20points%3D'5.5%2C0%202.5%2C3%201.1%2C3%204.1%2C0'%2F%3E%3Cpolygon%20points%3D'4%2C0%206%2C2%206%2C0.6%205.4%2C0'%2F%3E%3Cpolygon%20points%3D'0%2C2%201%2C3%202.4%2C3%200%2C0.6'%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E") repeat-x bottom left; }
.monaco-editor .squiggly-info { background: url("data:image/svg+xml,%3Csvg%20xmlns%3D'http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg'%20viewBox%3D'0%200%206%203'%20enable-background%3D'new%200%200%206%203'%20height%3D'3'%20width%3D'6'%3E%3Cg%20fill%3D'%233794ff'%3E%3Cpolygon%20points%3D'5.5%2C0%202.5%2C3%201.1%2C3%204.1%2C0'%2F%3E%3Cpolygon%20points%3D'4%2C0%206%2C2%206%2C0.6%205.4%2C0'%2F%3E%3Cpolygon%20points%3D'0%2C2%201%2C3%202.4%2C3%200%2C0.6'%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E") repeat-x bottom left; }
.monaco-editor .squiggly-hint { background: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20height%3D%223%22%20width%3D%2212%22%3E%3Cg%20fill%3D%22rgba(238%2C%20238%2C%20238%2C%200.7)%22%3E%3Ccircle%20cx%3D%221%22%20cy%3D%221%22%20r%3D%221%22%2F%3E%3Ccircle%20cx%3D%225%22%20cy%3D%221%22%20r%3D%221%22%2F%3E%3Ccircle%20cx%3D%229%22%20cy%3D%221%22%20r%3D%221%22%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E") no-repeat bottom left; }
.monaco-editor.showUnused .squiggly-inline-unnecessary { opacity: 0.667; }
.monaco-editor .selectionHighlight { background-color: rgba(173, 214, 255, 0.07); }

	.monaco-editor .diagonal-fill {
		background-image: linear-gradient(
			-45deg,
			rgba(204, 204, 204, 0.2) 12.5%,
			#0000 12.5%, #0000 50%,
			rgba(204, 204, 204, 0.2) 50%, rgba(204, 204, 204, 0.2) 62.5%,
			#0000 62.5%, #0000 100%
		);
		background-size: 8px 8px;
	}
	
.monaco-editor .findMatch { background-color: rgba(234, 92, 0, 0.33); }
.monaco-editor .currentFindMatch { background-color: #515c6a; }
.monaco-editor .findScope { background-color: rgba(58, 61, 65, 0.4); }
.monaco-editor .find-widget { background-color: #252526; }
.monaco-editor .find-widget { box-shadow: 0 0 8px 2px rgba(0, 0, 0, 0.36); }
.monaco-editor .find-widget { color: #cccccc; }
.monaco-editor .find-widget.no-results .matchesCount { color: #f48771; }
.monaco-editor .find-widget .monaco-sash { background-color: #454545; }

		.monaco-editor .find-widget .button:not(.disabled):hover,
		.monaco-editor .find-widget .codicon-find-selection:hover {
			background-color: rgba(90, 93, 94, 0.31) !important;
		}
	
.monaco-editor .find-widget .monaco-inputbox.synthetic-focus { outline-color: #007fd4; }
.monaco-editor .monaco-hover .hover-row:not(:first-child):not(:empty) { border-top: 1px solid rgba(69, 69, 69, 0.5); }
.monaco-editor .monaco-hover hr { border-top: 1px solid rgba(69, 69, 69, 0.5); }
.monaco-editor .monaco-hover hr { border-bottom: 0px solid rgba(69, 69, 69, 0.5); }
.monaco-editor { --vscode-foreground: #cccccc;
--vscode-disabledForeground: rgba(204, 204, 204, 0.5);
--vscode-errorForeground: #f48771;
--vscode-descriptionForeground: rgba(204, 204, 204, 0.7);
--vscode-icon-foreground: #c5c5c5;
--vscode-focusBorder: #007fd4;
--vscode-textSeparator-foreground: rgba(255, 255, 255, 0.18);
--vscode-textLink-foreground: #3794ff;
--vscode-textLink-activeForeground: #3794ff;
--vscode-textPreformat-foreground: #d7ba7d;
--vscode-textBlockQuote-background: rgba(127, 127, 127, 0.1);
--vscode-textBlockQuote-border: rgba(0, 122, 204, 0.5);
--vscode-textCodeBlock-background: rgba(10, 10, 10, 0.4);
--vscode-widget-shadow: rgba(0, 0, 0, 0.36);
--vscode-input-background: #3c3c3c;
--vscode-input-foreground: #cccccc;
--vscode-inputOption-activeBorder: #007acc;
--vscode-inputOption-hoverBackground: rgba(90, 93, 94, 0.5);
--vscode-inputOption-activeBackground: rgba(0, 127, 212, 0.4);
--vscode-inputOption-activeForeground: #ffffff;
--vscode-input-placeholderForeground: rgba(204, 204, 204, 0.5);
--vscode-inputValidation-infoBackground: #063b49;
--vscode-inputValidation-infoBorder: #007acc;
--vscode-inputValidation-warningBackground: #352a05;
--vscode-inputValidation-warningBorder: #b89500;
--vscode-inputValidation-errorBackground: #5a1d1d;
--vscode-inputValidation-errorBorder: #be1100;
--vscode-dropdown-background: #3c3c3c;
--vscode-dropdown-foreground: #f0f0f0;
--vscode-dropdown-border: #3c3c3c;
--vscode-button-foreground: #ffffff;
--vscode-button-separator: rgba(255, 255, 255, 0.4);
--vscode-button-background: #0e639c;
--vscode-button-hoverBackground: #1177bb;
--vscode-button-secondaryForeground: #ffffff;
--vscode-button-secondaryBackground: #3a3d41;
--vscode-button-secondaryHoverBackground: #45494e;
--vscode-badge-background: #4d4d4d;
--vscode-badge-foreground: #ffffff;
--vscode-scrollbar-shadow: #000000;
--vscode-scrollbarSlider-background: rgba(121, 121, 121, 0.4);
--vscode-scrollbarSlider-hoverBackground: rgba(100, 100, 100, 0.7);
--vscode-scrollbarSlider-activeBackground: rgba(191, 191, 191, 0.4);
--vscode-progressBar-background: #0e70c0;
--vscode-editorError-foreground: #f14c4c;
--vscode-editorWarning-foreground: #cca700;
--vscode-editorInfo-foreground: #3794ff;
--vscode-editorHint-foreground: rgba(238, 238, 238, 0.7);
--vscode-sash-hoverBorder: #007fd4;
--vscode-editor-background: #1e1e1e;
--vscode-editor-foreground: #d4d4d4;
--vscode-editorStickyScroll-background: #1e1e1e;
--vscode-editorStickyScrollHover-background: #2a2d2e;
--vscode-editorWidget-background: #252526;
--vscode-editorWidget-foreground: #cccccc;
--vscode-editorWidget-border: #454545;
--vscode-quickInput-background: #252526;
--vscode-quickInput-foreground: #cccccc;
--vscode-quickInputTitle-background: rgba(255, 255, 255, 0.1);
--vscode-pickerGroup-foreground: #3794ff;
--vscode-pickerGroup-border: #3f3f46;
--vscode-keybindingLabel-background: rgba(128, 128, 128, 0.17);
--vscode-keybindingLabel-foreground: #cccccc;
--vscode-keybindingLabel-border: rgba(51, 51, 51, 0.6);
--vscode-keybindingLabel-bottomBorder: rgba(68, 68, 68, 0.6);
--vscode-editor-selectionBackground: #264f78;
--vscode-editor-inactiveSelectionBackground: #3a3d41;
--vscode-editor-selectionHighlightBackground: rgba(173, 214, 255, 0.15);
--vscode-editor-findMatchBackground: #515c6a;
--vscode-editor-findMatchHighlightBackground: rgba(234, 92, 0, 0.33);
--vscode-editor-findRangeHighlightBackground: rgba(58, 61, 65, 0.4);
--vscode-searchEditor-findMatchBackground: rgba(234, 92, 0, 0.22);
--vscode-search-resultsInfoForeground: rgba(204, 204, 204, 0.65);
--vscode-editor-hoverHighlightBackground: rgba(38, 79, 120, 0.25);
--vscode-editorHoverWidget-background: #252526;
--vscode-editorHoverWidget-foreground: #cccccc;
--vscode-editorHoverWidget-border: #454545;
--vscode-editorHoverWidget-statusBarBackground: #2c2c2d;
--vscode-editorLink-activeForeground: #4e94ce;
--vscode-editorInlayHint-foreground: #cccccc;
--vscode-editorInlayHint-background: rgba(77, 77, 77, 0.25);
--vscode-editorInlayHint-typeForeground: #cccccc;
--vscode-editorInlayHint-typeBackground: rgba(77, 77, 77, 0.25);
--vscode-editorInlayHint-parameterForeground: #cccccc;
--vscode-editorInlayHint-parameterBackground: rgba(77, 77, 77, 0.25);
--vscode-editorLightBulb-foreground: #ffcc00;
--vscode-editorLightBulbAutoFix-foreground: #75beff;
--vscode-diffEditor-insertedTextBackground: rgba(156, 204, 44, 0.13);
--vscode-diffEditor-removedTextBackground: rgba(255, 0, 0, 0.14);
--vscode-diffEditor-insertedLineBackground: rgba(156, 204, 44, 0.13);
--vscode-diffEditor-removedLineBackground: rgba(255, 0, 0, 0.14);
--vscode-diffEditor-diagonalFill: rgba(204, 204, 204, 0.2);
--vscode-diffEditor-unchangedRegionBackground: #3e3e3e;
--vscode-diffEditor-unchangedRegionForeground: #a3a2a2;
--vscode-diffEditor-unchangedCodeBackground: rgba(116, 116, 116, 0.16);
--vscode-list-focusOutline: #007fd4;
--vscode-list-activeSelectionBackground: #04395e;
--vscode-list-activeSelectionForeground: #ffffff;
--vscode-list-inactiveSelectionBackground: #37373d;
--vscode-list-hoverBackground: #2a2d2e;
--vscode-list-dropBackground: #062f4a;
--vscode-list-highlightForeground: #2aaaff;
--vscode-list-focusHighlightForeground: #2aaaff;
--vscode-list-invalidItemForeground: #b89500;
--vscode-list-errorForeground: #f88070;
--vscode-list-warningForeground: #cca700;
--vscode-listFilterWidget-background: #252526;
--vscode-listFilterWidget-outline: rgba(0, 0, 0, 0);
--vscode-listFilterWidget-noMatchesOutline: #be1100;
--vscode-listFilterWidget-shadow: rgba(0, 0, 0, 0.36);
--vscode-list-filterMatchBackground: rgba(234, 92, 0, 0.33);
--vscode-tree-indentGuidesStroke: #585858;
--vscode-tree-inactiveIndentGuidesStroke: rgba(88, 88, 88, 0.4);
--vscode-tree-tableColumnsBorder: rgba(204, 204, 204, 0.13);
--vscode-tree-tableOddRowsBackground: rgba(204, 204, 204, 0.04);
--vscode-list-deemphasizedForeground: #8c8c8c;
--vscode-checkbox-background: #3c3c3c;
--vscode-checkbox-selectBackground: #252526;
--vscode-checkbox-foreground: #f0f0f0;
--vscode-checkbox-border: #3c3c3c;
--vscode-checkbox-selectBorder: #c5c5c5;
--vscode-quickInputList-focusForeground: #ffffff;
--vscode-quickInputList-focusBackground: #04395e;
--vscode-menu-foreground: #f0f0f0;
--vscode-menu-background: #3c3c3c;
--vscode-menu-selectionForeground: #ffffff;
--vscode-menu-selectionBackground: #04395e;
--vscode-menu-separatorBackground: #606060;
--vscode-toolbar-hoverBackground: rgba(90, 93, 94, 0.31);
--vscode-toolbar-activeBackground: rgba(99, 102, 103, 0.31);
--vscode-editor-snippetTabstopHighlightBackground: rgba(124, 124, 124, 0.3);
--vscode-editor-snippetFinalTabstopHighlightBorder: #525252;
--vscode-breadcrumb-foreground: rgba(204, 204, 204, 0.8);
--vscode-breadcrumb-background: #1e1e1e;
--vscode-breadcrumb-focusForeground: #e0e0e0;
--vscode-breadcrumb-activeSelectionForeground: #e0e0e0;
--vscode-breadcrumbPicker-background: #252526;
--vscode-merge-currentHeaderBackground: rgba(64, 200, 174, 0.5);
--vscode-merge-currentContentBackground: rgba(64, 200, 174, 0.2);
--vscode-merge-incomingHeaderBackground: rgba(64, 166, 255, 0.5);
--vscode-merge-incomingContentBackground: rgba(64, 166, 255, 0.2);
--vscode-merge-commonHeaderBackground: rgba(96, 96, 96, 0.4);
--vscode-merge-commonContentBackground: rgba(96, 96, 96, 0.16);
--vscode-editorOverviewRuler-currentContentForeground: rgba(64, 200, 174, 0.5);
--vscode-editorOverviewRuler-incomingContentForeground: rgba(64, 166, 255, 0.5);
--vscode-editorOverviewRuler-commonContentForeground: rgba(96, 96, 96, 0.4);
--vscode-editorOverviewRuler-findMatchForeground: rgba(209, 134, 22, 0.49);
--vscode-editorOverviewRuler-selectionHighlightForeground: rgba(160, 160, 160, 0.8);
--vscode-minimap-findMatchHighlight: #d18616;
--vscode-minimap-selectionOccurrenceHighlight: #676767;
--vscode-minimap-selectionHighlight: #264f78;
--vscode-minimap-errorHighlight: rgba(255, 18, 18, 0.7);
--vscode-minimap-warningHighlight: #cca700;
--vscode-minimap-foregroundOpacity: #000000;
--vscode-minimapSlider-background: rgba(121, 121, 121, 0.2);
--vscode-minimapSlider-hoverBackground: rgba(100, 100, 100, 0.35);
--vscode-minimapSlider-activeBackground: rgba(191, 191, 191, 0.2);
--vscode-problemsErrorIcon-foreground: #f14c4c;
--vscode-problemsWarningIcon-foreground: #cca700;
--vscode-problemsInfoIcon-foreground: #3794ff;
--vscode-charts-foreground: #cccccc;
--vscode-charts-lines: rgba(204, 204, 204, 0.5);
--vscode-charts-red: #f14c4c;
--vscode-charts-blue: #3794ff;
--vscode-charts-yellow: #cca700;
--vscode-charts-orange: #d18616;
--vscode-charts-green: #89d185;
--vscode-charts-purple: #b180d7;
--vscode-symbolIcon-arrayForeground: #cccccc;
--vscode-symbolIcon-booleanForeground: #cccccc;
--vscode-symbolIcon-classForeground: #ee9d28;
--vscode-symbolIcon-colorForeground: #cccccc;
--vscode-symbolIcon-constantForeground: #cccccc;
--vscode-symbolIcon-constructorForeground: #b180d7;
--vscode-symbolIcon-enumeratorForeground: #ee9d28;
--vscode-symbolIcon-enumeratorMemberForeground: #75beff;
--vscode-symbolIcon-eventForeground: #ee9d28;
--vscode-symbolIcon-fieldForeground: #75beff;
--vscode-symbolIcon-fileForeground: #cccccc;
--vscode-symbolIcon-folderForeground: #cccccc;
--vscode-symbolIcon-functionForeground: #b180d7;
--vscode-symbolIcon-interfaceForeground: #75beff;
--vscode-symbolIcon-keyForeground: #cccccc;
--vscode-symbolIcon-keywordForeground: #cccccc;
--vscode-symbolIcon-methodForeground: #b180d7;
--vscode-symbolIcon-moduleForeground: #cccccc;
--vscode-symbolIcon-namespaceForeground: #cccccc;
--vscode-symbolIcon-nullForeground: #cccccc;
--vscode-symbolIcon-numberForeground: #cccccc;
--vscode-symbolIcon-objectForeground: #cccccc;
--vscode-symbolIcon-operatorForeground: #cccccc;
--vscode-symbolIcon-packageForeground: #cccccc;
--vscode-symbolIcon-propertyForeground: #cccccc;
--vscode-symbolIcon-referenceForeground: #cccccc;
--vscode-symbolIcon-snippetForeground: #cccccc;
--vscode-symbolIcon-stringForeground: #cccccc;
--vscode-symbolIcon-structForeground: #cccccc;
--vscode-symbolIcon-textForeground: #cccccc;
--vscode-symbolIcon-typeParameterForeground: #cccccc;
--vscode-symbolIcon-unitForeground: #cccccc;
--vscode-symbolIcon-variableForeground: #75beff;
--vscode-editor-lineHighlightBorder: #282828;
--vscode-editor-rangeHighlightBackground: rgba(255, 255, 255, 0.04);
--vscode-editor-symbolHighlightBackground: rgba(234, 92, 0, 0.33);
--vscode-editorCursor-foreground: #aeafad;
--vscode-editorWhitespace-foreground: rgba(227, 228, 226, 0.16);
--vscode-editorIndentGuide-background: #404040;
--vscode-editorIndentGuide-activeBackground: #707070;
--vscode-editorLineNumber-foreground: #858585;
--vscode-editorActiveLineNumber-foreground: #c6c6c6;
--vscode-editorLineNumber-activeForeground: #c6c6c6;
--vscode-editorRuler-foreground: #5a5a5a;
--vscode-editorCodeLens-foreground: #999999;
--vscode-editorBracketMatch-background: rgba(0, 100, 0, 0.1);
--vscode-editorBracketMatch-border: #888888;
--vscode-editorOverviewRuler-border: rgba(127, 127, 127, 0.3);
--vscode-editorGutter-background: #1e1e1e;
--vscode-editorUnnecessaryCode-opacity: rgba(0, 0, 0, 0.67);
--vscode-editorGhostText-foreground: rgba(255, 255, 255, 0.34);
--vscode-editorOverviewRuler-rangeHighlightForeground: rgba(0, 122, 204, 0.6);
--vscode-editorOverviewRuler-errorForeground: rgba(255, 18, 18, 0.7);
--vscode-editorOverviewRuler-warningForeground: #cca700;
--vscode-editorOverviewRuler-infoForeground: #3794ff;
--vscode-editorBracketHighlight-foreground1: #ffd700;
--vscode-editorBracketHighlight-foreground2: #da70d6;
--vscode-editorBracketHighlight-foreground3: #179fff;
--vscode-editorBracketHighlight-foreground4: rgba(0, 0, 0, 0);
--vscode-editorBracketHighlight-foreground5: rgba(0, 0, 0, 0);
--vscode-editorBracketHighlight-foreground6: rgba(0, 0, 0, 0);
--vscode-editorBracketHighlight-unexpectedBracket-foreground: rgba(255, 18, 18, 0.8);
--vscode-editorBracketPairGuide-background1: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-background2: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-background3: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-background4: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-background5: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-background6: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-activeBackground1: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-activeBackground2: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-activeBackground3: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-activeBackground4: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-activeBackground5: rgba(0, 0, 0, 0);
--vscode-editorBracketPairGuide-activeBackground6: rgba(0, 0, 0, 0);
--vscode-editorUnicodeHighlight-border: #bd9b03;
--vscode-editorUnicodeHighlight-background: rgba(189, 155, 3, 0.15);
--vscode-editorOverviewRuler-bracketMatchForeground: #a0a0a0;
--vscode-editor-linkedEditingBackground: rgba(255, 0, 0, 0.3);
--vscode-editor-wordHighlightBackground: rgba(87, 87, 87, 0.72);
--vscode-editor-wordHighlightStrongBackground: rgba(0, 73, 114, 0.72);
--vscode-editor-wordHighlightTextBackground: rgba(87, 87, 87, 0.72);
--vscode-editorOverviewRuler-wordHighlightForeground: rgba(160, 160, 160, 0.8);
--vscode-editorOverviewRuler-wordHighlightStrongForeground: rgba(192, 160, 192, 0.8);
--vscode-editorOverviewRuler-wordHighlightTextForeground: rgba(160, 160, 160, 0.8);
--vscode-peekViewTitle-background: #252526;
--vscode-peekViewTitleLabel-foreground: #ffffff;
--vscode-peekViewTitleDescription-foreground: rgba(204, 204, 204, 0.7);
--vscode-peekView-border: #3794ff;
--vscode-peekViewResult-background: #252526;
--vscode-peekViewResult-lineForeground: #bbbbbb;
--vscode-peekViewResult-fileForeground: #ffffff;
--vscode-peekViewResult-selectionBackground: rgba(51, 153, 255, 0.2);
--vscode-peekViewResult-selectionForeground: #ffffff;
--vscode-peekViewEditor-background: #001f33;
--vscode-peekViewEditorGutter-background: #001f33;
--vscode-peekViewEditorStickyScroll-background: #001f33;
--vscode-peekViewResult-matchHighlightBackground: rgba(234, 92, 0, 0.3);
--vscode-peekViewEditor-matchHighlightBackground: rgba(255, 143, 0, 0.6);
--vscode-editorMarkerNavigationError-background: #f14c4c;
--vscode-editorMarkerNavigationError-headerBackground: rgba(241, 76, 76, 0.1);
--vscode-editorMarkerNavigationWarning-background: #cca700;
--vscode-editorMarkerNavigationWarning-headerBackground: rgba(204, 167, 0, 0.1);
--vscode-editorMarkerNavigationInfo-background: #3794ff;
--vscode-editorMarkerNavigationInfo-headerBackground: rgba(55, 148, 255, 0.1);
--vscode-editorMarkerNavigation-background: #1e1e1e;
--vscode-editorHoverWidget-highlightForeground: #2aaaff;
--vscode-editorSuggestWidget-background: #252526;
--vscode-editorSuggestWidget-border: #454545;
--vscode-editorSuggestWidget-foreground: #d4d4d4;
--vscode-editorSuggestWidget-selectedForeground: #ffffff;
--vscode-editorSuggestWidget-selectedBackground: #04395e;
--vscode-editorSuggestWidget-highlightForeground: #2aaaff;
--vscode-editorSuggestWidget-focusHighlightForeground: #2aaaff;
--vscode-editorSuggestWidgetStatus-foreground: rgba(212, 212, 212, 0.5);
--vscode-editor-foldBackground: rgba(38, 79, 120, 0.3);
--vscode-editorGutter-foldingControlForeground: #c5c5c5; }

.mtk1 { color: #d4d4d4; }
.mtk2 { color: #1e1e1e; }
.mtk3 { color: #cc6666; }
.mtk4 { color: #9cdcfe; }
.mtk5 { color: #ce9178; }
.mtk6 { color: #b5cea8; }
.mtk7 { color: #608b4e; }
.mtk8 { color: #6aa94f; }
.mtk9 { color: #569cd6; }
.mtk10 { color: #f28b82; }
.mtk11 { color: #d7ba7d; }
.mtk12 { color: #dcdcdc; }
.mtk13 { color: #808080; }
.mtk14 { color: #4ec9b0; }
.mtk15 { color: #dcdcaa; }
.mtk16 { color: #f44747; }
.mtk17 { color: #82c6ff; }
.mtk18 { color: #c586c0; }
.mtk19 { color: #a79873; }
.mtk20 { color: #dd6a6f; }
.mtk21 { color: #5bb498; }
.mtk22 { color: #909090; }
.mtk23 { color: #778899; }
.mtk24 { color: #ff00ff; }
.mtk25 { color: #b46695; }
.mtk26 { color: #ff0000; }
.mtk27 { color: #4f76ac; }
.mtk28 { color: #3dc9b0; }
.mtk29 { color: #74b0df; }
.mtk30 { color: #4864aa; }
.mtki { font-style: italic; }
.mtkb { font-weight: bold; }
.mtku { text-decoration: underline; text-underline-position: under; }
.mtks { text-decoration: line-through; }
.mtks.mtku { text-decoration: underline line-through; text-underline-position: under; }</style><style type="text/css">.MJXp-script {font-size: .8em}
.MJXp-right {-webkit-transform-origin: right; -moz-transform-origin: right; -ms-transform-origin: right; -o-transform-origin: right; transform-origin: right}
.MJXp-bold {font-weight: bold}
.MJXp-italic {font-style: italic}
.MJXp-scr {font-family: MathJax_Script,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-frak {font-family: MathJax_Fraktur,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-sf {font-family: MathJax_SansSerif,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-cal {font-family: MathJax_Caligraphic,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-mono {font-family: MathJax_Typewriter,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-largeop {font-size: 150%}
.MJXp-largeop.MJXp-int {vertical-align: -.2em}
.MJXp-math {display: inline-block; line-height: 1.2; text-indent: 0; font-family: 'Times New Roman',Times,STIXGeneral,serif; white-space: nowrap; border-collapse: collapse}
.MJXp-display {display: block; text-align: center; margin: 1em 0}
.MJXp-math span {display: inline-block}
.MJXp-box {display: block!important; text-align: center}
.MJXp-box:after {content: " "}
.MJXp-rule {display: block!important; margin-top: .1em}
.MJXp-char {display: block!important}
.MJXp-mo {margin: 0 .15em}
.MJXp-mfrac {margin: 0 .125em; vertical-align: .25em}
.MJXp-denom {display: inline-table!important; width: 100%}
.MJXp-denom > * {display: table-row!important}
.MJXp-surd {vertical-align: top}
.MJXp-surd > * {display: block!important}
.MJXp-script-box > *  {display: table!important; height: 50%}
.MJXp-script-box > * > * {display: table-cell!important; vertical-align: top}
.MJXp-script-box > *:last-child > * {vertical-align: bottom}
.MJXp-script-box > * > * > * {display: block!important}
.MJXp-mphantom {visibility: hidden}
.MJXp-munderover, .MJXp-munder {display: inline-table!important}
.MJXp-over {display: inline-block!important; text-align: center}
.MJXp-over > * {display: block!important}
.MJXp-munderover > *, .MJXp-munder > * {display: table-row!important}
.MJXp-mtable {vertical-align: .25em; margin: 0 .125em}
.MJXp-mtable > * {display: inline-table!important; vertical-align: middle}
.MJXp-mtr {display: table-row!important}
.MJXp-mtd {display: table-cell!important; text-align: center; padding: .5em 0 0 .5em}
.MJXp-mtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-mlabeledtr {display: table-row!important}
.MJXp-mlabeledtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mlabeledtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MJXp-scale0 {-webkit-transform: scaleX(.0); -moz-transform: scaleX(.0); -ms-transform: scaleX(.0); -o-transform: scaleX(.0); transform: scaleX(.0)}
.MJXp-scale1 {-webkit-transform: scaleX(.1); -moz-transform: scaleX(.1); -ms-transform: scaleX(.1); -o-transform: scaleX(.1); transform: scaleX(.1)}
.MJXp-scale2 {-webkit-transform: scaleX(.2); -moz-transform: scaleX(.2); -ms-transform: scaleX(.2); -o-transform: scaleX(.2); transform: scaleX(.2)}
.MJXp-scale3 {-webkit-transform: scaleX(.3); -moz-transform: scaleX(.3); -ms-transform: scaleX(.3); -o-transform: scaleX(.3); transform: scaleX(.3)}
.MJXp-scale4 {-webkit-transform: scaleX(.4); -moz-transform: scaleX(.4); -ms-transform: scaleX(.4); -o-transform: scaleX(.4); transform: scaleX(.4)}
.MJXp-scale5 {-webkit-transform: scaleX(.5); -moz-transform: scaleX(.5); -ms-transform: scaleX(.5); -o-transform: scaleX(.5); transform: scaleX(.5)}
.MJXp-scale6 {-webkit-transform: scaleX(.6); -moz-transform: scaleX(.6); -ms-transform: scaleX(.6); -o-transform: scaleX(.6); transform: scaleX(.6)}
.MJXp-scale7 {-webkit-transform: scaleX(.7); -moz-transform: scaleX(.7); -ms-transform: scaleX(.7); -o-transform: scaleX(.7); transform: scaleX(.7)}
.MJXp-scale8 {-webkit-transform: scaleX(.8); -moz-transform: scaleX(.8); -ms-transform: scaleX(.8); -o-transform: scaleX(.8); transform: scaleX(.8)}
.MJXp-scale9 {-webkit-transform: scaleX(.9); -moz-transform: scaleX(.9); -ms-transform: scaleX(.9); -o-transform: scaleX(.9); transform: scaleX(.9)}
.MathJax_PHTML .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><style type="text/css">.MathJax_Display {text-align: center; margin: 0; position: relative; display: block!important; text-indent: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; width: 100%}
.MathJax .merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MathJax .MJX-monospace {font-family: monospace}
.MathJax .MJX-sans-serif {font-family: sans-serif}
#MathJax_Tooltip {background-color: InfoBackground; color: InfoText; border: 1px solid black; box-shadow: 2px 2px 5px #AAAAAA; -webkit-box-shadow: 2px 2px 5px #AAAAAA; -moz-box-shadow: 2px 2px 5px #AAAAAA; -khtml-box-shadow: 2px 2px 5px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true'); padding: 3px 4px; z-index: 401; position: absolute; left: 0; top: 0; width: auto; height: auto; display: none}
.MathJax {display: inline; font-style: normal; font-weight: normal; line-height: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; padding: 0; margin: 0}
.MathJax:focus, body :focus .MathJax {display: inline-table}
.MathJax.MathJax_FullWidth {text-align: center; display: table-cell!important; width: 10000em!important}
.MathJax img, .MathJax nobr, .MathJax a {border: 0; padding: 0; margin: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; vertical-align: 0; line-height: normal; text-decoration: none}
img.MathJax_strut {border: 0!important; padding: 0!important; margin: 0!important; vertical-align: 0!important}
.MathJax span {display: inline; position: static; border: 0; padding: 0; margin: 0; vertical-align: 0; line-height: normal; text-decoration: none; box-sizing: content-box}
.MathJax nobr {white-space: nowrap!important}
.MathJax img {display: inline!important; float: none!important}
.MathJax * {transition: none; -webkit-transition: none; -moz-transition: none; -ms-transition: none; -o-transition: none}
.MathJax_Processing {visibility: hidden; position: fixed; width: 0; height: 0; overflow: hidden}
.MathJax_Processed {display: none!important}
.MathJax_test {font-style: normal; font-weight: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-transform: none; letter-spacing: normal; word-spacing: normal; overflow: hidden; height: 1px}
.MathJax_test.mjx-test-display {display: table!important}
.MathJax_test.mjx-test-inline {display: inline!important; margin-right: -1px}
.MathJax_test.mjx-test-default {display: block!important; clear: both}
.MathJax_ex_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60ex}
.MathJax_em_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60em}
.mjx-test-inline .MathJax_left_box {display: inline-block; width: 0; float: left}
.mjx-test-inline .MathJax_right_box {display: inline-block; width: 0; float: right}
.mjx-test-display .MathJax_right_box {display: table-cell!important; width: 10000em!important; min-width: 0; max-width: none; padding: 0; border: 0; margin: 0}
.MathJax .MathJax_HitBox {cursor: text; background: white; opacity: 0; filter: alpha(opacity=0)}
.MathJax .MathJax_HitBox * {filter: none; opacity: 1; background: transparent}
#MathJax_Tooltip * {filter: none; opacity: 1; background: transparent}
@font-face {font-family: MathJax_Main; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Main-Regular.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Main-Regular.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Main-bold; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Main-Bold.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Main-Bold.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Main-italic; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Main-Italic.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Main-Italic.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Math-italic; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Math-Italic.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Math-Italic.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Caligraphic; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Caligraphic-Regular.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Regular.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Size1; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Size1-Regular.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Size1-Regular.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Size2; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Size2-Regular.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Size2-Regular.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Size3; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Size3-Regular.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Size3-Regular.otf?V=2.7.5') format('opentype')}
@font-face {font-family: MathJax_Size4; src: url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/woff/MathJax_Size4-Regular.woff?V=2.7.5') format('woff'), url('https://colab.research.google.com/static/mathjax/fonts/HTML-CSS/TeX/otf/MathJax_Size4-Regular.otf?V=2.7.5') format('opentype')}
.MathJax .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><script async="async" type="text/javascript" src="./untitled10_files/python.js.download"></script></head><body class="" style="overscroll-behavior: contain;"><div style="visibility: hidden; overflow: hidden; position: absolute; top: 0px; height: 1px; width: auto; padding: 0px; border: 0px; margin: 0px; text-align: left; text-indent: 0px; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal;"><div id="MathJax_Hidden"></div></div><div id="MathJax_Message" style="display: none;"></div><div class="scripts"><script nonce="">window.performance.mark('external_scripts_start');</script><script src="./untitled10_files/gapi_loader.js.download" nonce=""></script><script src="./untitled10_files/socketio_binary.js.download" nonce=""></script><script src="./untitled10_files/analytics_binary.js.download" nonce=""></script><script src="./untitled10_files/MathJax.js.download" nonce=""></script><script src="./untitled10_files/js_monaco_editor_vs_loader.js.download" nonce=""></script><script nonce="">window.performance.mark('external_scripts_end'); window.performance.measure('external_scripts', 'external_scripts_start', 'external_scripts_end'); window.performance.mark('colab_load_start');
          window.Polymer = {
            'legacyOptimizations': true,
          };
          </script><script src="./untitled10_files/external_polymer_binary.js.download" nonce=""></script><script nonce="">
          window.performance.mark('colab_load_end');
          window.performance.measure('colab_load', 'colab_load_start', 'colab_load_end');
        </script></div><colab-snackbar leading="" labeltext="" id="message-area" class="message-area"><template shadowrootmode="open"><!----><style>
        :host .mdc-snackbar__surface {
          background-color: var(--colab-snackbar-surface-color);
        }
      </style>
      <!--?lit$2926288814$-->
      <div class="mdc-snackbar mdc-snackbar--leading">
        <div class="mdc-snackbar__surface">
          <!--?lit$2926288814$-->
          <div class="mdc-snackbar__actions">
            <slot name="action"></slot>
            <slot name="dismiss"></slot>
          </div>
        </div>
      </div><!--?--></template>
      <md-icon-button class="close" slot="dismiss" title="Close" aria-label="Close" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Close">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
        <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>close</md-icon>
      </md-icon-button>
    </colab-snackbar><colab-snackbar leading="" labeltext="" id="message-area-secondary" class="message-area"><template shadowrootmode="open"><!----><style>
        :host .mdc-snackbar__surface {
          background-color: var(--colab-snackbar-surface-color);
        }
      </style>
      <!--?lit$2926288814$-->
      <div class="mdc-snackbar mdc-snackbar--leading">
        <div class="mdc-snackbar__surface">
          <!--?lit$2926288814$--><div class="mdc-snackbar__label" role="status" aria-live="polite">Mounting Google Drive...</div>
          <div class="mdc-snackbar__actions">
            <slot name="action"></slot>
            <slot name="dismiss"></slot>
          </div>
        </div>
      </div><!--?--></template>
      <md-icon-button class="close" slot="dismiss" title="Close" aria-label="Close" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Close">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
        <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>close</md-icon>
      </md-icon-button>
    </colab-snackbar><div ng-non-bindable=""></div><div class="gb_s" ng-non-bindable=""><div class="gb_Hc"><div>Google Account</div><div class="gb_Hb">Lucky Champ</div><div>luckychamp141@gmail.com</div></div></div><script nonce="">this.gbar_=this.gbar_||{};(function(_){var window=this;
try{
var rd;_.od=function(a){var b=typeof a;return"object"==b&&null!=a||"function"==b};_.pd=function(a,b){if(void 0!==a.i||void 0!==a.j)throw Error("A");a.j=b;_.Lc(a)};_.qd=class extends _.Q{constructor(a){super(a)}};rd=class extends _.Zc{};_.sd=function(a,b){if(b in a.i)return a.i[b];throw new rd;};_.td=function(a){return _.sd(_.Wc.i(),a)};
}catch(e){_._DumpException(e)}
try{
/*

 SPDX-License-Identifier: Apache-2.0
*/
var zd,Id,Kd;_.ud=function(a){if(null==a)return a;if("string"===typeof a){if(!a)return;a=+a}if("number"===typeof a)return Number.isFinite(a)?a|0:void 0};_.vd=function(a){const b=a.length;if(0<b){const c=Array(b);for(let d=0;d<b;d++)c[d]=a[d];return c}return[]};_.xd=function(a){if(a instanceof _.wd)return a.i;throw Error("C");};zd=function(a){return new yd(b=>b.substr(0,a.length+1).toLowerCase()===a+":")};
_.Bd=function(a,b=_.Ad){if(a instanceof _.wd)return a;for(let c=0;c<b.length;++c){const d=b[c];if(d instanceof yd&&d.Yg(a))return new _.wd(a)}};_.Dd=function(a){if(Cd.test(a))return a};_.Ed=function(a){return a instanceof _.wd?_.xd(a):_.Dd(a)};_.Fd=function(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}};_.Gd=function(a,b,c){return void 0!==_.ib(a,b,c,!1)};_.Hd=function(a,b){return _.ud(_.rc(a,b))};
_.R=function(a,b){a=_.rc(a,b);return null==a?a:Number.isFinite(a)?a|0:void 0};_.S=function(a,b,c=0){return _.jb(_.Hd(a,b),c)};Id=0;_.Jd=function(a){return Object.prototype.hasOwnProperty.call(a,_.vb)&&a[_.vb]||(a[_.vb]=++Id)};Kd=function(a){return a};_.Ld=function(a){var b=null,c=_.q.trustedTypes;if(!c||!c.createPolicy)return b;try{b=c.createPolicy(a,{createHTML:Kd,createScript:Kd,createScriptURL:Kd})}catch(d){_.q.console&&_.q.console.error(d.message)}return b};
_.Md=function(a,b){return 0==a.lastIndexOf(b,0)};_.Nd=function(a,b){return Array.prototype.some.call(a,b,void 0)};try{(new self.OffscreenCanvas(0,0)).getContext("2d")}catch(a){};var Od;_.Pd=function(){void 0===Od&&(Od=_.Ld("ogb-qtm#html"));return Od};var Sd;_.Qd=class{constructor(a){this.i=a}toString(){return this.i+""}};_.Rd=function(a){return a instanceof _.Qd&&a.constructor===_.Qd?a.i:"type_error:TrustedResourceUrl"};Sd={};_.Td=function(a){const b=_.Pd();a=b?b.createScriptURL(a):a;return new _.Qd(a,Sd)};_.wd=class{constructor(a){this.i=a}toString(){return this.i}};_.Ud=new _.wd("about:invalid#zClosurez");var yd,Cd;yd=class{constructor(a){this.Yg=a}};_.Ad=[zd("data"),zd("http"),zd("https"),zd("mailto"),zd("ftp"),new yd(a=>/^[^:]*([/?#]|$)/.test(a))];Cd=/^\s*(?!javascript:)(?:[\w+.-]+:|[^:/?#]*(?:[/?#]|$))/i;_.Vd={};_.Wd=class{constructor(a){this.i=a}toString(){return this.i.toString()}};_.Xd=new _.Wd("",_.Vd);_.Yd=RegExp("^[-+,.\"'%_!#/ a-zA-Z0-9\\[\\]]+$");_.Zd=RegExp("\\b(url\\([ \t\n]*)('[ -&(-\\[\\]-~]*'|\"[ !#-\\[\\]-~]*\"|[!#-&*-\\[\\]-~]*)([ \t\n]*\\))","g");_.$d=RegExp("\\b(calc|cubic-bezier|fit-content|hsl|hsla|linear-gradient|matrix|minmax|radial-gradient|repeat|rgb|rgba|(rotate|scale|translate)(X|Y|Z|3d)?|steps|var)\\([-+*/0-9a-zA-Z.%#\\[\\], ]+\\)","g");var ae;ae={};_.ce=function(a){return a instanceof _.be&&a.constructor===_.be?a.i:"type_error:SafeHtml"};_.de=function(a){const b=_.Pd();a=b?b.createHTML(a):a;return new _.be(a,ae)};_.be=class{constructor(a){this.i=a}toString(){return this.i.toString()}};_.ee=new _.be(_.q.trustedTypes&&_.q.trustedTypes.emptyHTML||"",ae);_.fe=_.de("<br>");var he;_.ge=function(a){let b=!1,c;return function(){b||(c=a(),b=!0);return c}}(function(){var a=document.createElement("div"),b=document.createElement("div");b.appendChild(document.createElement("div"));a.appendChild(b);b=a.firstChild.firstChild;a.innerHTML=_.ce(_.ee);return!b.parentElement});he=/^[\w+/_-]+[=]{0,2}$/;
_.ie=function(a){a=(a||_.q).document;return a.querySelector?(a=a.querySelector('style[nonce],link[rel="stylesheet"][nonce]'))&&(a=a.nonce||a.getAttribute("nonce"))&&he.test(a)?a:"":""};_.je=function(a,b){this.width=a;this.height=b};_.m=_.je.prototype;_.m.aspectRatio=function(){return this.width/this.height};_.m.Eb=function(){return!(this.width*this.height)};_.m.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};_.m.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};_.m.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};_.T=function(a,b){var c=b||document;if(c.getElementsByClassName)a=c.getElementsByClassName(a)[0];else{c=document;var d=b||c;a=d.querySelectorAll&&d.querySelector&&a?d.querySelector(a?"."+a:""):_.ke(c,a,b)[0]||null}return a||null};
_.ke=function(a,b,c){var d;a=c||a;if(a.querySelectorAll&&a.querySelector&&b)return a.querySelectorAll(b?"."+b:"");if(b&&a.getElementsByClassName){var e=a.getElementsByClassName(b);return e}e=a.getElementsByTagName("*");if(b){var f={};for(c=d=0;a=e[c];c++){var g=a.className;"function"==typeof g.split&&_.sa(g.split(/\s+/),b)&&(f[d++]=a)}f.length=d;return f}return e};_.me=function(a){return _.le(document,a)};
_.le=function(a,b){b=String(b);"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());return a.createElement(b)};_.ne=function(a){for(var b;b=a.firstChild;)a.removeChild(b)};_.oe=function(a){return 9==a.nodeType?a:a.ownerDocument||a.document};
}catch(e){_._DumpException(e)}
try{
_.Wi=function(a){var b;let c;const d=null==(c=(b=(a.ownerDocument&&a.ownerDocument.defaultView||window).document).querySelector)?void 0:c.call(b,"script[nonce]");(b=d?d.nonce||d.getAttribute("nonce")||"":"")&&a.setAttribute("nonce",b)};_.Xi=function(a){if(!a)return null;a=_.G(a,4);var b;null===a||void 0===a?b=null:b=_.Td(a);return b};_.Yi=class extends _.Q{constructor(a){super(a)}};_.Zi=function(a,b){return(b||document).getElementsByTagName(String(a))};
}catch(e){_._DumpException(e)}
try{
var aj=function(a,b,c){a<b?$i(a+1,b):_.Ec.log(Error("ba`"+a+"`"+b),{url:c})},$i=function(a,b){if(bj){const c=_.me("SCRIPT");c.async=!0;c.type="text/javascript";c.charset="UTF-8";c.src=_.Rd(bj);_.Wi(c);c.onerror=_.Fd(aj,a,b,c.src);_.Zi("HEAD")[0].appendChild(c)}},cj=class extends _.Q{constructor(a){super(a)}};var dj=_.E(_.Qc,cj,17)||new cj,ej,bj=(ej=_.E(dj,_.Yi,1))?_.Xi(ej):null,fj,gj=(fj=_.E(dj,_.Yi,2))?_.Xi(fj):null,hj=function(){$i(1,2);if(gj){const b=_.me("LINK");b.setAttribute("type","text/css");b.rel="stylesheet";b.href=_.Rd(gj).toString();var a=_.ie(b.ownerDocument&&b.ownerDocument.defaultView);a&&b.setAttribute("nonce",a);(a=_.ie())&&b.setAttribute("nonce",a);_.Zi("HEAD")[0].appendChild(b)}};(function(){const a=_.Rc();if(_.D(a,18))hj();else{const b=_.Hd(a,19)||0;window.addEventListener("load",()=>{window.setTimeout(hj,b)})}})();
}catch(e){_._DumpException(e)}
})(this.gbar_);
// Google Inc.
</script><div style="position: absolute; width: 0px; height: 0px; overflow: hidden; padding: 0px; border: 0px; margin: 0px;"><div id="MathJax_Font_Test" style="position: absolute; visibility: hidden; top: 0px; left: 0px; width: auto; min-width: 0px; max-width: none; padding: 0px; border: 0px; margin: 0px; white-space: nowrap; text-align: left; text-indent: 0px; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; font-size: 40px; font-weight: normal; font-style: normal; font-family: MathJax_Size1, monospace;"></div></div><div class="notebook-vertical colab-left-pane-open" style="position: relative;">
      <div class="top-floater"><div role="banner">
    <colab-header-skip-button><template shadowrootmode="open"><!----><a id="skiplink" class="skip-link" href="https://colab.research.google.com/#top-toolbar"><!--?lit$2926288814$-->Skip to main content</a></template></colab-header-skip-button>
    <!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
          <div id="private-outputs-warning" class="header-warning private-outputs-warning hidden">
            <!--?lit$2926288814$-->This notebook is open with private outputs. Outputs will not be saved. You can disable this in <a href="https://colab.research.google.com/#" id="private-outputs-notebook-info-link" command="notebook-settings" aria-describedby="private-outputs-notebook-info-link-tooltip">Notebook settings</a><colab-tooltip-trigger aria-hidden="true" for="private-outputs-notebook-info-link" id="private-outputs-notebook-info-link-tooltip"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Open notebook settings</div><!----><!--?--></template>
        </colab-tooltip-trigger>.
          <mwc-icon-button class="close" icon="close" title="Close"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label="Close"><!--?lit$2926288814$-->
    <!--?lit$2926288814$--><i class="material-icons"><!--?lit$2926288814$-->close</i>
    <span><slot></slot></span>
  </button></template></mwc-icon-button></div>
        

    <div id="header" class="horizontal layout" style="display: none;">
      <div id="header-background"><div></div></div>
      <div id="header-content">
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><div id="header-logo">
              <!--?lit$2926288814$--> <!--?lit$2926288814$--><a href="https://drive.google.com/drive/search?q=owner%3Ame%20(type%3Aapplication%2Fvnd.google.colaboratory%20%7C%7C%20type%3Aapplication%2Fvnd.google.colab)&amp;authuser=0" aria-label="View in Google Drive">
        <!--?lit$2926288814$--><md-icon class="colab-large-icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$-->
      <g id="colab-logo">
        <path d="M4.54,9.46,2.19,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36A3.59,3.59,0,0,1,4.54,9.46Z" style="fill:var(--colab-logo-dark)"></path>
        <path d="M2.19,7.1,4.54,9.46a3.59,3.59,0,0,1,5.08,0l1.71-2.93h0l-.1-.08h0A6.93,6.93,0,0,0,2.19,7.1Z" style="fill:var(--colab-logo-light)"></path>
        <path d="M11.34,17.46h0L9.62,14.54a3.59,3.59,0,0,1-5.08,0L2.19,16.9a6.93,6.93,0,0,0,9,.65l.11-.09" style="fill:var(--colab-logo-light)"></path>
        <path d="M12,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36a3.59,3.59,0,1,1,5.08-5.08L21.81,7.1A6.93,6.93,0,0,0,12,7.1Z" style="fill:var(--colab-logo-light)"></path>
        <path d="M21.81,7.1,19.46,9.46a3.59,3.59,0,0,1-5.08,5.08L12,16.9A6.93,6.93,0,0,0,21.81,7.1Z" style="fill:var(--colab-logo-dark)"></path>
      </g></svg></md-icon>
      </a><!--?-->
            </div>
        <div id="header-doc-toolbar" class="flex">
          <div id="document-info">
            <!--?lit$2926288814$--> <!--?lit$2926288814$-->
    <input id="doc-name" class="doc-name" maxlength="259" autocomplete="off" aria-label="Notebook name" disabled="" style="width: auto;"><colab-input-sizer aria-hidden="true" style="left: -1000%; position: absolute; font-family: &quot;Google Sans&quot;, Roboto, Noto, sans-serif; font-size: 18px; font-weight: 400; letter-spacing: normal; padding-left: 3px; padding-right: 4px; white-space: pre;">Welcome To Colab_</colab-input-sizer>
            <!--?lit$2926288814$-->
          </div>
        <div class="menubar-wrapper"><div><!----><div id="top-menubar" class="goog-menubar format-lightborder" role="menubar" tabindex="0" style="user-select: none;"><!--?lit$2926288814$--><div class="goog-menu-button goog-inline-block" id="file-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->File</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div><div class="goog-menu-button goog-inline-block" id="edit-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->Edit</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div><div class="goog-menu-button goog-inline-block" id="view-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->View</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div><div class="goog-menu-button goog-inline-block" id="insert-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->Insert</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div><div class="goog-menu-button goog-inline-block" id="runtime-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->Runtime</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div><div class="goog-menu-button goog-inline-block" id="tools-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->Tools</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div><div class="goog-menu-button goog-inline-block" id="help-menu-button" role="button" aria-expanded="false" aria-haspopup="true" style="user-select: none;"><div class="goog-inline-block goog-menu-button-outer-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-inner-box" style="user-select: none;"><div class="goog-inline-block goog-menu-button-caption" style="user-select: none;"><!--?lit$2926288814$-->Help</div><div class="goog-inline-block goog-menu-button-dropdown" style="user-select: none;">&nbsp;</div></div></div></div></div>
    <div id="colab-menu-cover" style="display: none;"> </div></div><!----><colab-last-saved-indicator aria-live="polite" aria-atomic="true" title="You do not have permission to save this notebook. To keep your changes, make a copy of the notebook."><template shadowrootmode="open"><!----><button class=" save-message "><!--?lit$2926288814$-->Cannot save changes</button></template></colab-last-saved-indicator></div></div>
        <div id="header-right">
          <!--?lit$2926288814$-->
    <colab-collaborator-bar id="collaborator-bar"><template shadowrootmode="open"><!----> <div class="collaborator-bar">
      <!--?lit$2926288814$-->
      <!--?lit$2926288814$-->
    </div></template></colab-collaborator-bar>
  
          <!--?lit$2926288814$-->
          <!--?lit$2926288814$--> <md-text-button id="share-toolbar-button" command="share" role="presentation" aria-describedby="share-toolbar-button-tooltip" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$-->
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
                <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->link</md-icon>
                <!--?lit$2926288814$-->Share
              </md-text-button><colab-tooltip-trigger aria-hidden="true" for="share-toolbar-button" id="share-toolbar-button-tooltip"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Share notebook</div><!----><!--?--></template>
        </colab-tooltip-trigger>
          <!--?lit$2926288814$--> <md-icon-button id="settings-cog" command="preferences" title="Open settings" aria-label="Open settings" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Open settings">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
                <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>settings</md-icon>
              </md-icon-button>
          <div class="header-onegoogle-container"><div class="onegoogle"><div class="gb_Ra gb_md gb_ib gb_2c" id="gb"><div class="gb_Cd gb_gb gb_rd" ng-non-bindable="" data-ogsr-up="" style="padding:0;height:auto;display:block"><div class="gb_Vd" style="display:block"><div class="gb_8c"></div><div class="gb_b gb_y gb_4f gb_L"><div class="gb_f gb_fb gb_4f gb_L"><a class="gb_d gb_Ia gb_L" aria-label="Google Account: Lucky Champ  
(luckychamp141@gmail.com)" href="https://accounts.google.com/SignOutOptions?hl=en&amp;continue=https://colab.research.google.com/&amp;ec=GBRAqQM" tabindex="0" role="button"><img class="gb_p gbii" src="./untitled10_files/unnamed.png" srcset="https://lh3.google.com/u/0/ogw/AF2bZyhLIJmuTpshn1nIq_u_vwY4kpvW-Ug7Bp8OUGBzZR_6Tw=s32-c-mo 1x, https://lh3.google.com/u/0/ogw/AF2bZyhLIJmuTpshn1nIq_u_vwY4kpvW-Ug7Bp8OUGBzZR_6Tw=s64-c-mo 2x " alt="" aria-hidden="true" data-noaft=""></a></div></div></div><div style="overflow: hidden; position: absolute; top: 0px; visibility: hidden; width: 436px; z-index: 991; height: 0px; margin-top: 57px; right: 0px; margin-right: 4px;"></div></div></div><script nonce="">this.gbar_=this.gbar_||{};(function(_){var window=this;
try{
_.kd=function(a,b,c){if(!a.j)if(c instanceof Array)for(var d of c)_.kd(a,b,d);else{d=(0,_.y)(a.C,a,b);const e=a.v+c;a.v++;b.dataset.eqid=e;a.B[e]=d;b&&b.addEventListener?b.addEventListener(c,d,!1):b&&b.attachEvent?b.attachEvent("on"+c,d):a.o.log(Error("y`"+b))}};
}catch(e){_._DumpException(e)}
try{
_.ld=function(){if(!_.q.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});try{const c=()=>{};_.q.addEventListener("test",c,b);_.q.removeEventListener("test",c,b)}catch(c){}return a}();
}catch(e){_._DumpException(e)}
try{
var md=document.querySelector(".gb_k .gb_d"),nd=document.querySelector("#gb.gb_Zc");md&&!nd&&_.kd(_.Vc,md,"click");
}catch(e){_._DumpException(e)}
try{
_.qh=function(a){const b=[];let c=0;for(const d in a)b[c++]=a[d];return b};_.rh=function(a){if(a.o)return a.o;for(const b in a.i)if(a.i[b].na()&&a.i[b].B())return a.i[b];return null};_.sh=function(a,b){a.i[b.K()]=b};var th=new class extends _.Cc{constructor(){var a=_.Ec;super();this.B=a;this.o=null;this.j={};this.C={};this.i={};this.v=null}A(a){this.i[a]&&(_.rh(this)&&_.rh(this).K()==a||this.i[a].O(!0))}Wa(a){this.v=a;for(const b in this.i)this.i[b].na()&&this.i[b].Wa(a)}qc(a){return a in this.i?this.i[a]:null}};_.Yc("dd",th);
}catch(e){_._DumpException(e)}
try{
_.Pi=function(a,b){return _.K(a,36,b)};
}catch(e){_._DumpException(e)}
try{
var Qi=document.querySelector(".gb_b .gb_d"),Ri=document.querySelector("#gb.gb_Zc");Qi&&!Ri&&_.kd(_.Vc,Qi,"click");
}catch(e){_._DumpException(e)}
})(this.gbar_);
// Google Inc.
</script></div></div>
        </div>
      </div>
    </div>
  </div></div><div class="notebook-horizontal">
        <!--?lit$2926288814$--><colab-left-pane role="complementary" aria-label="left pane" class="colab-left-pane-open"><!----><div class="colab-left-pane-nib layout vertical" role="toolbar" aria-orientation="vertical">
        <div class="left-pane-top"><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button toggle="" command="show-toc-pane" aria-label="Table of contents" title="Table of contents" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button standard" aria-label="Table of contents" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_bulleted</md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button toggle="" command="find" aria-label="Find and replace" title="Find and replace" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Find and replace" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->search</md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button toggle="" command="show-variables" aria-label="Variables" title="Variables" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Variables" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$-->
      <path d="M4.51,9.44V6.08c0-1.34.37-1.85,1.6-2.17l.22-.06V3.13l-.27,0-.44,0a4.46,4.46,0,0,0-2.2.59,2.78,2.78,0,0,0-1,2.51V9.74c0,1.26-.26,1.61-1.49,2L0,12l.94.29c1.21.38,1.49.75,1.49,2v3.5a2.94,2.94,0,0,0,1,2.6,4.39,4.39,0,0,0,2.14.56l.46,0,.27,0v-.72l-.22-.06c-1.24-.32-1.6-.81-1.6-2.17V14.58c0-1.43-.3-2.13-1.25-2.57C4.2,11.57,4.51,10.87,4.51,9.44Z"></path>
      <path d="M23.06,11.71c-1.22-.36-1.49-.71-1.49-2l0-3.5a3,3,0,0,0-1-2.6,4.38,4.38,0,0,0-2.14-.56l-.46,0-.27,0v.72l.22.06c1.24.32,1.6.81,1.6,2.17V9.44c0,1.44.3,2.13,1.25,2.57-1,.44-1.25,1.14-1.25,2.57v3.36c0,1.34-.37,1.85-1.6,2.17l-.22.06v.72l.27,0,.44,0a4.47,4.47,0,0,0,2.2-.59,2.82,2.82,0,0,0,1-2.51V14.28c0-1.26.26-1.61,1.49-2L24,12Z"></path>
      <path d="M15.16,8.22a.88.88,0,0,1,.46.16,1.25,1.25,0,0,0,.69.2h0A1,1,0,0,0,17,8.23a1.06,1.06,0,0,0,.24-.8,1.1,1.1,0,0,0-1.15-1h0c-1,0-1.73.64-3,2.57l-.12-.51c-.28-1.36-.56-2-1.39-2h0A8,8,0,0,0,9,7.08l-.47.16.16.91L9.41,8a3.22,3.22,0,0,1,.73-.14c.34,0,.43,0,.71,1.2l.56,2.47L9.76,13.82a3.6,3.6,0,0,1-.8.88.9.9,0,0,1-.38-.13,1.83,1.83,0,0,0-.88-.28,1,1,0,0,0-1,1.06A1.15,1.15,0,0,0,8,16.53c.85,0,1.35-.35,2.24-1.55l1.49-2,.46,1.88c.23,1,.46,1.66,1.53,1.66s1.66-.75,2.81-2.53l.17-.26-.81-.48-.16.2-.25.34-.19.25c-.45.57-.62.73-.76.73s-.28-.4-.34-.63l-.67-2.83a4.2,4.2,0,0,1-.15-.79C13.84,9.78,14.74,8.22,15.16,8.22Z"></path></svg></md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button toggle="" command="open-user-secrets" aria-label="Secrets" title="Secrets" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Secrets" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->vpn_key</md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button toggle="" command="show-files" aria-label="Files" title="Files" role="presentation" value="" selected=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button standard selected" aria-label="Files" aria-pressed="true">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="icon icon--selected"><slot name="selected"><slot></slot></slot></span>
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->folder</md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div></div>
        <div class="left-pane-bottom"><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button command="snippets" aria-label="Code snippets" title="Code snippets" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Code snippets">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->code</md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button command="show-command-palette" aria-label="Command palette" title="Command palette" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Command palette">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$-->
      <path d="M21,3H3A2,2,0,0,0,1,5V17a2,2,0,0,0,2,2H21a2,2,0,0,0,2-2V5A2,2,0,0,0,21,3Zm0,2V17H3V5"></path>
      <rect x="5" y="12" width="11" height="2"></rect>
      <rect x="5" y="8" width="11" height="2"></rect>
      <rect x="17" y="8" width="2" height="2"></rect>
      <rect x="17" y="12" width="2" height="2"></rect></svg></md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div><!----><div class="left-pane-button">
        <!--?lit$2926288814$--><md-icon-button command="show-terminal" aria-label="Terminal" title="Terminal" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Terminal">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->terminal</md-icon>
        </md-icon-button> <!--?lit$2926288814$-->
      </div></div>
      </div><colab-resizer class="ew-resize">
          <div class="resizer-contents">
            <div class="colab-left-pane-header layout horizontal noshrink">
              <h3 class="left-pane-content-title">Files</h3>
              <!--?lit$2926288814$--><md-icon-button class="colab-left-pane-move" aria-label="Move left pane to a tab" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Move left pane to a tab">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
      <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>tab</md-icon>
    </md-icon-button> <!--?lit$2926288814$--><md-icon-button class="colab-left-pane-close" aria-label="Close left pane" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Close left pane">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
      <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>close</md-icon>
    </md-icon-button>
            </div>
            <div class="left-pane-container"><colab-file-browser class="layout vertical"><colab-file-tree><!----> <div class="file-tree-buttons">
          <label for="file-tree-upload-input">
            <md-icon-button class="colab-icon file-tree-upload-button" title="Upload to session storage" aria-label="Upload to session storage" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Upload to session storage">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
              <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>upload_file</md-icon>
            </md-icon-button>
          </label>
          <input id="file-tree-upload-input" type="file" multiple="">
          <md-icon-button class="file-tree-refresh" title="Refresh" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard ">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
            <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>refresh</md-icon>
          </md-icon-button>
          <md-icon-button toggle="" class="mount-drive-button" title="Mount Drive" aria-label="Mount Drive" aria-label-selected="Unmount Drive" role="presentation" value="" style="display: flex;"><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Mount Drive" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
            <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$-->
    <path d="M20,6H12L10,4H4A2,2,0,0,0,2,6V18a2,2,0,0,0,2,2H20a2,2,0,0,0,2-2V8A2,2,0,0,0,20,6ZM13.57,9.32,13,8.4l.34-.6h3.11l3.69,6.5H16.38M12.11,17l-.67,1.2h-1L9,15.42,12.72,9l2,3.46h0M19.3,18.2H12.09l.67-1.2,1.15-2h6.64l.34.6Z"></path></svg></md-icon>
            <md-icon slot="selected" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$-->
    <path d="M12.72,9l2,3.46h0l-.26.47,2,2h4.1l.34.6L19.45,18l1.69,1.69A2,2,0,0,0,22,18V8a2,2,0,0,0-2-2H12L10,4H5.5l6.4,6.4Zm.66-1.17h3.11l3.69,6.5H16.38l-2.81-5L13,8.4Z"></path>
    <path d="M22.19,22.19,1.81,1.81.4,3.22,2.25,5.07A2,2,0,0,0,2,6V18a2,2,0,0,0,2,2H17.17l3.61,3.61Zm-11.73-4L9,15.42l1.3-2.26,2.35,2.36.17.17L12.11,17l-.67,1.2Zm1.63,0,.67-1.2.51-.9,2.1,2.1Z"></path></svg></md-icon>
          </md-icon-button>
          <md-icon-button toggle="" id="hidden-files-toggle" aria-label="Show hidden files" aria-label-selected="Hide hidden files" role="presentation" title="Show hidden files" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Show hidden files" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
            <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>visibility_off</md-icon>
            <md-icon slot="selected" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>visibility</md-icon>
          </md-icon-button>
        </div>
        <div class="parent-link"><div class="file-title-row">
          <!--?lit$2926288814$-->
                <div class="file-icon colab-icon" style="margin-left: 0px;"></div>
              
          <md-icon class="file-icon colab-icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->folder_open</md-icon>
          <span class="file-tree-name" title="Up one level">
            <!--?lit$2926288814$-->..
          </span>
          <input class="file-tree-name-input" value="..">
          <md-icon-button class="file-item-menu" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard ">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_vert</md-icon></md-icon-button>
        </div></div>
        <div class="files-root"><colab-file-view filename="content" draggable="true"><!----><!--?lit$2926288814$--><!--?--><!--?lit$2926288814$-->
        <div class="child-files"><colab-file-view filename="sample_data" class="collapsed" draggable="true"><!----><!--?lit$2926288814$-->
        <div class="file-title-row">
          <!--?lit$2926288814$--> <md-icon class="file-icon colab-icon directory-icon" style="margin-left: 0px;" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>arrow_drop_down</md-icon>
          <md-icon class="file-icon colab-icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->folder</md-icon>
          <span class="file-tree-name" title="sample_data">
            <!--?lit$2926288814$-->sample_data
          </span>
          <input class="file-tree-name-input" value="sample_data">
          <md-icon-button class="file-item-menu" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard ">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_vert</md-icon></md-icon-button>
        </div>
      <!--?lit$2926288814$-->
        <div class="child-files"></div>
        <div class="overflow-ellipsis" title="Directory is too large to display all files." style="margin-left: 0px;">
          …
        </div>
      <!--?--></colab-file-view><colab-file-view filename="diabetes.csv" draggable="true"><!----><!--?lit$2926288814$-->
        <div class="file-title-row">
          <!--?lit$2926288814$-->
                <div class="file-icon colab-icon" style="margin-left: 0px;"></div>
              
          <md-icon class="file-icon colab-icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->draft</md-icon>
          <span class="file-tree-name" title="diabetes.csv (23.31K) 
Last modified Wed Apr 24 2024 23:06:47 GMT-0500 (Central Daylight Time)">
            <!--?lit$2926288814$-->diabetes.csv
          </span>
          <input class="file-tree-name-input" value="diabetes.csv">
          <md-icon-button class="file-item-menu" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard ">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_vert</md-icon></md-icon-button>
        </div>
      <!--?lit$2926288814$--><!--?--><!--?--></colab-file-view></div>
        <div class="overflow-ellipsis" title="Directory is too large to display all files." style="margin-left: -20px;">
          …
        </div>
      <!--?--></colab-file-view></div>
        <div class="files-drag-to-upload layout horizontal">
          <md-icon class="layout noshrink" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>upload_file</md-icon>
          <div> <!--?lit$2926288814$-->Drop files to upload them to session storage. </div>
        </div>
        <div class="files-uploading"></div>
        <div class="colab-usage-bar-container"><colab-usage-bar id="file-browser-disk-display-kernel" class="file-browser-disk-display usage-bar-with-suffix" style="display: flex;"><template shadowrootmode="open"><!---->
      <div class="label"><!--?lit$2926288814$-->Disk </div>
      <div class="total">
        <div usage="low" style="width:24%;"> </div>
      </div>
      <!--?lit$2926288814$--><div class="suffix"><!--?lit$2926288814$-->81.52 GB available</div>
    </template></colab-usage-bar><colab-tooltip-trigger aria-hidden="true" id="file-browser-disk-display-kernel-tooltip"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Disk: 26.20 GB/107.72 GB</div><!----><!--?--></template></colab-tooltip-trigger></div></colab-file-tree></colab-file-browser></div>
          </div>
        <div class="resizer-thumb"></div></colab-resizer></colab-left-pane>
        <div class="layout vertical grow">
          <colab-notebook-toolbar id="top-toolbar" class="horizontal layout center noshrink collapsed"><!----> <!--?lit$2926288814$-->
          <colab-toolbar-button command="insert-cell-below" icon="add" id="toolbar-add-code"><template shadowrootmode="open"><!----><md-text-button id="button" aria-disabled="false" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$-->
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
        <!--?lit$2926288814$--><md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->add</md-icon>
        <span class="button-content"><slot></slot></span>
        <!--?lit$2926288814$--><span class="screenreader-only"><!--?lit$2926288814$-->Insert code cell below <!--?lit$2926288814$-->Ctrl+M B</span>
      </md-text-button>
      <!--?lit$2926288814$--> <colab-tooltip-trigger for="button" aria-hidden="true" id="tooltip" message="Insert code cell below" shortcut="Ctrl+M B"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Insert code cell below</div><!----><!--?--></template>
          </colab-tooltip-trigger><!--?--></template>
            <!--?lit$2926288814$-->Code
          </colab-toolbar-button>
          <colab-toolbar-button command="add-text" icon="add" id="toolbar-add-text"><template shadowrootmode="open"><!----><md-text-button id="button" aria-disabled="false" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$-->
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
        <!--?lit$2926288814$--><md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->add</md-icon>
        <span class="button-content"><slot></slot></span>
        <!--?lit$2926288814$--><span class="screenreader-only"><!--?lit$2926288814$-->Add text cell <!--?lit$2926288814$--></span>
      </md-text-button>
      <!--?lit$2926288814$--> <colab-tooltip-trigger for="button" aria-hidden="true" id="tooltip" message="Add text cell" shortcut=""><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Add text cell</div><!----><!--?--></template>
          </colab-tooltip-trigger><!--?--></template>
            <!--?lit$2926288814$-->Text
          </colab-toolbar-button>
          <!--?lit$2926288814$-->
        
    <!--?lit$2926288814$-->
    <!--?lit$2926288814$--> <span class="expanded-options">
          <span class="colab-separator"></span>
          <colab-toolbar-button command="clone"><template shadowrootmode="open"><!----><md-text-button id="button" aria-disabled="false" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$-->
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
        <!--?lit$2926288814$-->
        <span class="button-content"><slot></slot></span>
        <!--?lit$2926288814$--><span class="screenreader-only"><!--?lit$2926288814$--> <!--?lit$2926288814$--></span>
      </md-text-button>
      <!--?lit$2926288814$--><!--?--></template>
            <!--?lit$2926288814$-->Copy to Drive
          </colab-toolbar-button>
        </span>
    <!--?lit$2926288814$-->
    <!--?lit$2926288814$--> <span class="collapsed-options">
          <colab-last-saved-indicator aria-live="polite" aria-atomic="true" title="You do not have permission to save this notebook. To keep your changes, make a copy of the notebook."><template shadowrootmode="open"><!----><button class=" save-message "><!--?lit$2926288814$-->Cannot save changes</button></template></colab-last-saved-indicator>
        </span>

    <span class="flex"></span>
    <!--?lit$2926288814$--><colab-connect-warning-button><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!--?--><!--?--></template></colab-connect-warning-button>
    <!--?lit$2926288814$--><!--?lit$2926288814$--><colab-connect-button><template shadowrootmode="open"><!----> <!--?lit$2926288814$--><md-icon-button id="connect-icon" class="big-icon icon-okay" aria-label="Focus the last run cell" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Focus the last run cell">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
            <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->done</md-icon>
          </md-icon-button>
          <colab-tooltip-trigger for="connect-icon" id="connect-icon-tooltip" aria-hidden="true" message="Focus the last run cell"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Focus the last run cell</div><!----><!--?--></template>
          </colab-tooltip-trigger>
      <colab-toolbar-button id="connect" tooltipid="colab-connect-tooltip" tooltiptext="Connected to

          Python 3 Google Compute Engine backend

          

RAM: 1.09 GB/12.67 GB
Disk: 26.20 GB/107.72 GB"><template shadowrootmode="open"><!----><md-text-button id="button" aria-disabled="false" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$-->
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
        <!--?lit$2926288814$-->
        <span class="button-content"><slot></slot></span>
        <!--?lit$2926288814$--><span class="screenreader-only"><!--?lit$2926288814$-->Connected to

          Python 3 Google Compute Engine backend

          

RAM: 1.09 GB/12.67 GB
Disk: 26.20 GB/107.72 GB <!--?lit$2926288814$--></span>
      </md-text-button>
      <!--?lit$2926288814$--> <colab-tooltip-trigger for="button" aria-hidden="true" id="colab-connect-tooltip" message="Connected to

          Python 3 Google Compute Engine backend

          

RAM: 1.09 GB/12.67 GB
Disk: 26.20 GB/107.72 GB" shortcut=""><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Connected to</div><!----><!----><div><!--?lit$2926288814$--></div><!----><!----><div><!--?lit$2926288814$-->          Python 3 Google Compute Engine backend</div><!----><!----><div><!--?lit$2926288814$--></div><!----><!----><div><!--?lit$2926288814$-->          </div><!----><!----><div><!--?lit$2926288814$--></div><!----><!----><div><!--?lit$2926288814$-->RAM: 1.09 GB/12.67 GB</div><!----><!----><div><!--?lit$2926288814$-->Disk: 26.20 GB/107.72 GB</div><!----><!--?--></template>
          </colab-tooltip-trigger><!--?--></template>
        <!--?lit$2926288814$--> <div id="connect-button-resource-display">
        <!--?lit$2926288814$--><colab-usage-sparkline class="ram" label="RAM"><template shadowrootmode="open"><!---->
      <div class="label"><!--?lit$2926288814$-->RAM</div>
      <!--?lit$2926288814$-->
      <canvas height="14" width="20"></canvas>
    </template></colab-usage-sparkline>
        <!--?lit$2926288814$--><colab-usage-sparkline class="disks" label="Disk"><template shadowrootmode="open"><!---->
      <div class="label"><!--?lit$2926288814$-->Disk</div>
      <!--?lit$2926288814$-->
      <canvas height="14" width="20"></canvas>
    </template></colab-usage-sparkline>
      </div>
      </colab-toolbar-button>
      <!--?lit$2926288814$--> <md-icon-button id="connect-dropdown" aria-label="Additional connection options" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Additional connection options">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
        <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>arrow_drop_down</md-icon>
      </md-icon-button>
      <colab-tooltip-trigger for="connect-dropdown" id="connect-dropdown-tooltip" aria-hidden="true" message="Additional connection options"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Additional connection options</div><!----><!--?--></template>
      </colab-tooltip-trigger><!--?--></template></colab-connect-button><!--?-->
    <!--?lit$2926288814$--> <span class="colab-separator"></span>
          <colab-toolbar-button icon="supervised_user_circle" command="show-chat"><template shadowrootmode="open"><!----><md-text-button id="button" aria-disabled="false" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$-->
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
        <!--?lit$2926288814$--><md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->supervised_user_circle</md-icon>
        <span class="button-content"><slot></slot></span>
        <!--?lit$2926288814$--><span class="screenreader-only"><!--?lit$2926288814$--> <!--?lit$2926288814$--></span>
      </md-text-button>
      <!--?lit$2926288814$--><!--?--></template>
            <!--?lit$2926288814$-->Colab AI
          </colab-toolbar-button>
    <!--?lit$2926288814$-->
    <span class="collapsed-options">
      <!--?lit$2926288814$--><span class="colab-separator"></span>
      <!--?lit$2926288814$--> <md-icon-button command="share" title="Share notebook" aria-label="Share notebook" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Share notebook">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
            <md-icon filled="" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->link</md-icon>
          </md-icon-button><md-icon-button command="preferences" aria-label="Open settings" title="Open settings" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Open settings">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
        <md-icon filled="" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>settings</md-icon>
      </md-icon-button>
    </span>
    <span class="colab-separator"></span>
    <!--?lit$2926288814$--><md-icon-button toggle="" command="toggle-header" id="toggle-header-button" aria-label="Toggle header visibility" role="presentation" aria-describedby="toggle-header-button-tooltip" selected="" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  selected standard " aria-label="Toggle header visibility" aria-pressed="true">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="icon icon--selected"><slot name="selected"><slot></slot></slot></span>
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
    <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>expand_less</md-icon>
    <md-icon slot="selected" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>expand_more</md-icon>
  </md-icon-button><colab-tooltip-trigger aria-hidden="true" for="toggle-header-button" id="toggle-header-button-tooltip"><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----><div><!--?lit$2926288814$-->Toggle header visibility</div><!----><!--?--></template>
        </colab-tooltip-trigger><!--?--></colab-notebook-toolbar><colab-tab-layout-container class="layout horizontal grow flexible-tabs"><!----> <div class="layout horizontal tab-pane-parent">
      <!--?lit$2926288814$--> <div class="layout vertical tab-pane-parent">
      <!--?lit$2926288814$--><colab-tab-pane class="layout vertical grow no-header focused" align="horizontal"><!----> <div class="layout vertical grow">
    <div class="tab-pane-header layout horizontal noshrink">
      <md-tabs><template shadowrootmode="open"><!---->
      <div class="tabs">
        <slot></slot>
      </div>
      <md-divider part="divider"><template shadowrootmode="open"><!----></template></md-divider>
    </template><md-primary-tab noink="" title="" aria-labelledby="tab-title-QwhRDYBgbXuT" class="selected-tab" md-tab="" active="" tabindex="0"><template shadowrootmode="open"><!----><div class="button" role="presentation">
      <md-focus-ring part="focus-ring" inward="" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-elevation part="elevation" aria-hidden="true"><template shadowrootmode="open"><!----><span class="shadow"></span></template></md-elevation>
      <md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <div role="presentation" class="content  has-label stacked ">
        <slot name="icon"></slot>
        <slot></slot>
        <!--?lit$2926288814$--><div class="indicator"></div>
      </div>
      <!--?lit$2926288814$-->
    </div></template>
          <div class="colab-tab-header">
            <span class="colab-tab-title" id="tab-title-QwhRDYBgbXuT"><!--?lit$2926288814$--><!--?lit$2926288814$-->Notebook<!--?--></span>
            <!--?lit$2926288814$-->
          </div>
        </md-primary-tab></md-tabs>
      <div class="layout grow"></div>
      <!--?lit$2926288814$--> <md-icon-button title="Show more" aria-label="Show more" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Show more">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
    <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_horiz</md-icon>
  </md-icon-button>
    </div>
    <div class="layout vertical grow tab-pane-container"> <colab-tab class="layout vertical grow notebook-tab-content selected-tab"><!----> <div class="overflow-flexbox-workaround">
      <colab-shaded-scroller ignore-dom-changes="" role="main" class="notebook-container" aria-label="Notebook" tabindex="-1">
        <div class="notebook-scrolling-horizontal-container">
          <div class="notebook-scrolling-horizontal">
            <div class="notebook-content-background">
              <!--?lit$2926288814$-->
              <div class="notebook-content ">
                <!--?lit$2926288814$--><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div>
                <div class="notebook-cell-list"><div class="cell code icon-scrolling code-has-output" id="cell-0n356ZQ9bZUg" role="region" aria-label="Cell 0: Code cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><div class="codecell-input-output">
      <div class="inputarea horizontal layout code">
        <div class="cell-gutter">
          <!-- Bounding range for vertical scrolling of icons -->
          <div class="cell-execution-container">
            <colab-run-button title="Run cell (Ctrl+Enter)
cell executed since last change

executed at 11:06 PM (6 minutes ago)
executed in 13.071s" role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution">
      <div class="execution-count"><!--?lit$2926288814$-->[1]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><div id="status">
      <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></svg></md-icon>
      <div><!--?lit$2926288814$-->13s</div>
    </div>
    </div></template></colab-run-button>
          </div>
        </div>
      <div class="editor flex lazy-editor" style=""><div class="editor flex monaco" data-keybinding-context="47" data-mode-id="notebook-python" style="height: 29px; --vscode-editorCodeLens-lineHeight: 16px; --vscode-editorCodeLens-fontSize: 12px; --vscode-editorCodeLens-fontFeatureSettings: &quot;liga&quot; off, &quot;calt&quot; off;"><div class="monaco-editor no-user-select  showUnused showDeprecated vs-dark" role="code" data-uri="inmemory://model/19" style="width: 1174px; height: 29px;"><div data-mprt="3" class="overflow-guard" style="width: 1174px; height: 29px;"><div class="margin" role="presentation" aria-hidden="true" style="position: absolute; contain: strict; will-change: unset; top: 0px; height: 29px; width: 6px;"><div class="glyph-margin" style="left: 0px; width: 0px; height: 29px;"></div><div class="margin-view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="margin-view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 6px; height: 29px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line current-line-margin-both" style="width:6px; height:19px;"></div></div></div><div class="glyph-margin-widgets" style="position: absolute; top: 0px;"></div></div><div class="monaco-scrollable-element editor-scrollable vs-dark" role="presentation" data-mprt="5" style="position: absolute; overflow: hidden; left: 6px; width: 1168px; height: 29px;"><div class="lines-content monaco-editor-background" style="position: absolute; overflow: hidden; width: 1e+06px; height: 29px; contain: strict; will-change: unset; top: 0px; left: 0px;"><div class="view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; height: 0px; width: 1168px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line" style="width:1168px; height:19px;"></div><div class="cdr squiggly-warning" style="left:0px;width:23px;height:19px;"></div><div class="cdr squiggly-warning" style="left:31px;width:54px;height:19px;"></div><div class="cdr squiggly-warning" style="left:92px;width:46px;height:19px;"></div><div class="cdr squiggly-warning" style="left:146px;width:77px;height:19px;"></div><div class="cdr squiggly-error" style="left:31px;width:54px;height:19px;"></div><div class="cdr squiggly-error" style="left:92px;width:46px;height:19px;"></div><div class="cdr squiggly-error" style="left:146px;width:77px;height:19px;"></div></div></div><div role="presentation" aria-hidden="true" class="view-rulers"></div><div class="view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="view-lines monaco-mouse-cursor-text" role="presentation" aria-hidden="true" data-mprt="7" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 1168px; height: 29px;"><div style="top:0px;height:19px;" class="view-line"><span><span class="mtk1">pip&nbsp;install&nbsp;pandas&nbsp;matplotlib</span></span></div></div><div data-mprt="1" class="contentWidgets" style="position: absolute; top: 0px;"><div class="lightBulbWidget codicon codicon-light-bulb" widgetid="LightBulbWidget" title="Show Code Actions (Ctrl+.)" style="position: absolute; display: none; visibility: hidden; max-width: 1168px;"></div></div><div role="presentation" aria-hidden="true" class="cursors-layer cursor-line-style cursor-solid"><div class="cursor monaco-mouse-cursor-text " style="height: 19px; top: 0px; left: 0px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; display: block; visibility: hidden; padding-left: 0px; width: 1.66667px;"></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute; width: 1154px; height: 10px; left: 0px; bottom: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; width: 1154px;"></div></div><canvas class="decorationsOverviewRuler" aria-hidden="true" width="16" height="34" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; top: 0px; right: 0px; width: 14px; height: 29px; will-change: unset; display: block;"></canvas><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute; width: 14px; height: 29px; right: 0px; top: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 14px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; height: 29px;"></div></div></div><div role="presentation" aria-hidden="true" style="width: 1174px;"></div><textarea data-mprt="6" class="inputarea monaco-mouse-cursor-text" wrap="on" autocorrect="off" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Editor content;Press Alt+F1 for Accessibility Options." tabindex="0" role="textbox" aria-roledescription="editor" aria-multiline="true" aria-haspopup="false" aria-autocomplete="both" style="tab-size: 15.3984px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; top: 0px; left: 6px; width: 1174px; height: 1px;"></textarea><div class="monaco-editor-background textAreaCover" style="position: absolute; top: 0px; left: 0px; width: 0px; height: 0px;"></div><div data-mprt="4" class="overlayWidgets" style="width: 1174px;"><div widgetid="editor.contrib.quickInputWidget" style="position: absolute; top: 0px; right: 50%;"></div></div><div data-mprt="8" class="minimap slider-mouseover" role="presentation" aria-hidden="true" style="position: absolute; left: 0px; width: 0px; height: 29px;"><div class="minimap-shadow-hidden" style="height: 29px;"></div><canvas width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><canvas class="minimap-decorations-layer" width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><div class="minimap-slider" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; width: 0px; will-change: unset;"><div class="minimap-slider-horizontal" style="position: absolute; width: 0px; height: 0px;"></div></div></div><div role="presentation" aria-hidden="true" class="blockDecorations-container"></div></div><div data-mprt="2" class="overflowingContentWidgets"><div widgetid="editor.contrib.resizableContentHoverWidget" style="position: fixed; height: 10px; width: 10px; z-index: 50; display: none; visibility: hidden; max-width: 1600px;"><div class="monaco-sash vertical" style="left: 8px;"></div><div class="monaco-sash vertical" style="left: -2px;"></div><div class="monaco-sash orthogonal-edge-north horizontal" style="top: -2px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-sash orthogonal-edge-south horizontal" style="top: 8px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-hover hidden" role="tooltip" tabindex="0"><div class="monaco-scrollable-element " role="presentation" style="position: relative; overflow: hidden;"><div class="monaco-hover-content" style="overflow: hidden; font-size: 14px; line-height: 1.35714; max-width: 774.84px; max-height: 250px;"></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div class="shadow"></div><div class="shadow"></div><div class="shadow"></div></div></div></div></div><div class=".in-cell-overflowing"></div></div></div></div><colab-form class="formview vertical layout flex"><div class="widget-area vertical layout"></div></colab-form></div>
    <div class="output"><!----> <div class="output-header"> </div>
        <div class="output-content" style="">
          <div class="output-info"><colab-output-info title="Clear output

executed at 11:06 PM (0 minutes ago)
executed in 13.071s" hovering=""><template shadowrootmode="open"><!----> <md-icon class="collaborator account-circle" alt="Execution output" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->account_circle</md-icon>
      <!--?lit$2926288814$--> <mwc-icon-button icon="cancel" command="clear-focused-or-selected-outputs" alt="Clear output"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label="cancel"><!--?lit$2926288814$-->
    <!--?lit$2926288814$--><i class="material-icons"><!--?lit$2926288814$-->cancel</i>
    <span><slot></slot></span>
  </button></template>
          </mwc-icon-button><!--?--></template></colab-output-info></div>
          <div class="output-iframe-container">
            <div class="output-iframe-sizer" style="min-height: 0px;"> <div><div><colab-static-output-renderer role="group" tabindex="0"><div><div class="stream output-id-2 output_text"><pre>Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.0.3)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.7.1)
Requirement already satisfied: python-dateutil&gt;=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)
Requirement already satisfied: pytz&gt;=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2023.4)
Requirement already satisfied: tzdata&gt;=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.1)
Requirement already satisfied: numpy&gt;=1.21.0 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.25.2)
Requirement already satisfied: contourpy&gt;=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.2.1)
Requirement already satisfied: cycler&gt;=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools&gt;=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.51.0)
Requirement already satisfied: kiwisolver&gt;=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.5)
Requirement already satisfied: packaging&gt;=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.0)
Requirement already satisfied: pillow&gt;=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (9.4.0)
Requirement already satisfied: pyparsing&gt;=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (3.1.2)
Requirement already satisfied: six&gt;=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil&gt;=2.8.2-&gt;pandas) (1.16.0)
</pre></div></div><div></div></colab-static-output-renderer></div></div><div><div></div></div></div>
          </div>
        </div></div></div><colab-cell-next-steps><template shadowrootmode="open"><!----></template></colab-cell-next-steps></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell text" id="cell-TxzF4I6AchL8" role="region" aria-label="Cell 1: Text cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><!----> <div class="toolbar-root"><div class="markdown-toolbar"><!----><!--?lit$2926288814$--><!----><mwc-icon-button class="markdown-toolbar-header" title="Toggle heading"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_size</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-bold" title="Bold"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_bold</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-italic" title="Italicize"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_italic</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-code" title="Format as code"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->code</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-link" title="Insert link"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->link</md-icon>
  </mwc-icon-button><!----><!----><label for="markdown-image-input" class="colab-icon markdown-insert-image"><span role="button" title="Insert image" aria-label="Insert image" tabindex="0"><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>image</md-icon></span></label><input id="markdown-image-input" type="file" multiple="" accept="image/*"><!----><!----><mwc-icon-button class="markdown-toolbar-blockquote" title="Add blockquote"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_quote</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-ol" title="Add numbered list"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_numbered</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-ul" title="Add bulleted list"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_bulleted</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-hr" title="Add horizontal rule"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->horizontal_rule</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="latex" title="LaTeX"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><span>ψ</span>
  </mwc-icon-button><!----><!----><mwc-icon-button title="Insert emoji"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->mood</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-preview" title="Reposition markdown preview"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><g id="markdown-preview-below">
  <rect width="20" height="18" x="2" y="2" rx="2" ry="2" style="fill:none;stroke:var(--colab-icon-color)"></rect>
  <line x1="4.5" y1="13" x2="19.5" y2="13" style="stroke:var(--colab-primary-text-color);stroke-dasharray:2"></line>
  <line x1="2.5" y1="4" x2="21.5" y2="4" style="stroke:var(--colab-icon-color);stroke-width:3px;"></line>
</g></svg></md-icon>
  </mwc-icon-button><!----><!--?--></div></div>
      <div class="editor-container horizontal">
        <div class="editor-root"><div class="editor flex monaco" data-keybinding-context="16" data-mode-id="markdown" style="height: 29px; --vscode-editorCodeLens-lineHeight: 16px; --vscode-editorCodeLens-fontSize: 12px; --vscode-editorCodeLens-fontFeatureSettings: &quot;liga&quot; off, &quot;calt&quot; off; display: none;"><div class="monaco-editor no-user-select  showUnused showDeprecated vs-dark" role="code" data-uri="inmemory://model/24" style="width: 611px; height: 29px;"><div data-mprt="3" class="overflow-guard" style="width: 611px; height: 29px;"><div class="margin" role="presentation" aria-hidden="true" style="position: absolute; contain: strict; will-change: unset; top: 0px; height: 29px; width: 6px;"><div class="glyph-margin" style="left: 0px; width: 0px; height: 29px;"></div><div class="margin-view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="margin-view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 6px; height: 29px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line current-line-margin-both" style="width:6px; height:19px;"></div></div></div><div class="glyph-margin-widgets" style="position: absolute; top: 0px;"></div></div><div class="monaco-scrollable-element editor-scrollable vs-dark" role="presentation" data-mprt="5" style="position: absolute; overflow: hidden; left: 6px; width: 605px; height: 29px;"><div class="lines-content monaco-editor-background" style="position: absolute; overflow: hidden; width: 1e+06px; height: 29px; contain: strict; will-change: unset; top: 0px; left: 0px;"><div class="view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; height: 0px; width: 605px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line" style="width:605px; height:19px;"></div></div></div><div role="presentation" aria-hidden="true" class="view-rulers"></div><div class="view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="view-lines monaco-mouse-cursor-text" role="presentation" aria-hidden="true" data-mprt="7" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 605px; height: 29px;"><div style="top:0px;height:19px;" class="view-line"><span><span class="mtk1">a</span><span class="mtk1 unexpected-closing-bracket">)</span></span></div></div><div data-mprt="1" class="contentWidgets" style="position: absolute; top: 0px;"><div class="lightBulbWidget codicon codicon-light-bulb" widgetid="LightBulbWidget" title="Show Code Actions (Ctrl+.)" style="position: absolute; display: none; visibility: hidden; max-width: 605px;"></div></div><div role="presentation" aria-hidden="true" class="cursors-layer cursor-line-style cursor-solid"><div class="cursor monaco-mouse-cursor-text " style="height: 19px; top: 0px; left: 15px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; display: block; visibility: hidden; padding-left: 0px; width: 1.66667px;"></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute; width: 591px; height: 10px; left: 0px; bottom: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; width: 591px;"></div></div><canvas class="decorationsOverviewRuler" aria-hidden="true" width="16" height="34" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; top: 0px; right: 0px; width: 14px; height: 29px; will-change: unset; display: block;"></canvas><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute; width: 14px; height: 29px; right: 0px; top: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 14px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; height: 29px;"></div></div></div><div role="presentation" aria-hidden="true" style="width: 611px;"></div><textarea data-mprt="6" class="inputarea monaco-mouse-cursor-text" wrap="on" autocorrect="off" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Editor content;Press Alt+F1 for Accessibility Options." tabindex="0" role="textbox" aria-roledescription="editor" aria-multiline="true" aria-haspopup="false" aria-autocomplete="both" style="tab-size: 15.3984px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; top: 0px; left: 6px; width: 585px; height: 1px;"></textarea><div style="position: absolute; top: 0px; left: 0px; width: 0px; height: 0px;" class="monaco-editor-background textAreaCover"></div><div data-mprt="4" class="overlayWidgets" style="width: 611px;"></div><div data-mprt="8" class="minimap slider-mouseover" role="presentation" aria-hidden="true" style="position: absolute; left: 0px; width: 0px; height: 29px;"><div class="minimap-shadow-hidden" style="height: 29px;"></div><canvas width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><canvas class="minimap-decorations-layer" width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><div class="minimap-slider" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; width: 0px; will-change: unset;"><div class="minimap-slider-horizontal" style="position: absolute; width: 0px; height: 0px;"></div></div></div><div role="presentation" aria-hidden="true" class="blockDecorations-container"></div></div><div data-mprt="2" class="overflowingContentWidgets" style="display: none;"><div widgetid="editor.contrib.resizableContentHoverWidget" style="position: fixed; height: 10px; width: 10px; z-index: 50; display: none; visibility: hidden; max-width: 1600px;"><div class="monaco-sash vertical" style="left: 8px;"></div><div class="monaco-sash vertical" style="left: -2px;"></div><div class="monaco-sash orthogonal-edge-north horizontal" style="top: -2px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-sash orthogonal-edge-south horizontal" style="top: 8px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-hover hidden" role="tooltip" tabindex="0"><div class="monaco-scrollable-element " role="presentation" style="position: relative; overflow: hidden;"><div class="monaco-hover-content" style="overflow: hidden; font-size: 14px; line-height: 1.35714; max-width: 500px; max-height: 250px;"></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div class="shadow"></div><div class="shadow"></div><div class="shadow"></div></div></div></div></div><div class=".in-cell-overflowing"><div widgetid="editor.contrib.quickInputWidget" style="position: absolute; top: 0px; right: 50%;"></div></div></div></div></div>
        <div class="text-top-div">
          <div class="markdown"><span><p>a)</p>
</span></div>
        </div>
      </div>

      <div class="section-header" style="display: none;">
        <colab-run-button role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution">
      <div class="execution-count"><!--?lit$2926288814$-->[ ]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><!--?-->
    </div></template></colab-run-button>
        <div class="section-header-container" title="click to expand">↳ 0 cells hidden</div>
      </div></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell code icon-scrolling code-has-output" id="cell-9S3uN_AvsJ5p" role="region" aria-label="Cell 2: Code cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><div class="codecell-input-output">
      <div class="inputarea horizontal layout code">
        <div class="cell-gutter">
          <!-- Bounding range for vertical scrolling of icons -->
          <div class="cell-execution-container">
            <colab-run-button title="Run cell (Ctrl+Enter)
cell executed since last change

executed at 11:10 PM (2 minutes ago)
executed in 1.41s" role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution">
      <div class="execution-count"><!--?lit$2926288814$-->[5]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><div id="status">
      <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></svg></md-icon>
      <div><!--?lit$2926288814$-->1s</div>
    </div>
    </div></template></colab-run-button>
          </div>
        </div>
      <div class="editor flex lazy-editor" style=""><div class="editor flex monaco" data-keybinding-context="46" data-mode-id="notebook-python" style="height: 599px; --vscode-editorCodeLens-lineHeight: 16px; --vscode-editorCodeLens-fontSize: 12px; --vscode-editorCodeLens-fontFeatureSettings: &quot;liga&quot; off, &quot;calt&quot; off;"><div class="monaco-editor no-user-select  showUnused showDeprecated vs-dark" role="code" data-uri="inmemory://model/25" style="width: 1174px; height: 599px;"><div data-mprt="3" class="overflow-guard" style="width: 1174px; height: 599px;"><div class="margin" role="presentation" aria-hidden="true" style="position: absolute; contain: strict; will-change: unset; top: 0px; height: 599px; width: 6px;"><div class="glyph-margin" style="left: 0px; width: 0px; height: 599px;"></div><div class="margin-view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="margin-view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 6px; height: 599px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line current-line-margin-both" style="width:6px; height:19px;"></div></div><div style="position:absolute;top:19px;width:100%;height:19px;"></div><div style="position:absolute;top:38px;width:100%;height:19px;"></div><div style="position:absolute;top:57px;width:100%;height:19px;"></div><div style="position:absolute;top:76px;width:100%;height:19px;"></div><div style="position:absolute;top:95px;width:100%;height:19px;"></div><div style="position:absolute;top:114px;width:100%;height:19px;"></div><div style="position:absolute;top:133px;width:100%;height:19px;"></div><div style="position:absolute;top:152px;width:100%;height:19px;"></div><div style="position:absolute;top:171px;width:100%;height:19px;"></div><div style="position:absolute;top:190px;width:100%;height:19px;"></div><div style="position:absolute;top:209px;width:100%;height:19px;"></div><div style="position:absolute;top:228px;width:100%;height:19px;"></div><div style="position:absolute;top:247px;width:100%;height:19px;"></div><div style="position:absolute;top:266px;width:100%;height:19px;"></div><div style="position:absolute;top:285px;width:100%;height:19px;"></div><div style="position:absolute;top:304px;width:100%;height:19px;"></div><div style="position:absolute;top:323px;width:100%;height:19px;"></div><div style="position:absolute;top:342px;width:100%;height:19px;"></div><div style="position:absolute;top:361px;width:100%;height:19px;"></div><div style="position:absolute;top:380px;width:100%;height:19px;"></div><div style="position:absolute;top:399px;width:100%;height:19px;"></div><div style="position:absolute;top:418px;width:100%;height:19px;"></div><div style="position:absolute;top:437px;width:100%;height:19px;"></div><div style="position:absolute;top:456px;width:100%;height:19px;"></div><div style="position:absolute;top:475px;width:100%;height:19px;"></div><div style="position:absolute;top:494px;width:100%;height:19px;"></div><div style="position:absolute;top:513px;width:100%;height:19px;"></div><div style="position:absolute;top:532px;width:100%;height:19px;"></div><div style="position:absolute;top:551px;width:100%;height:19px;"></div><div style="position:absolute;top:570px;width:100%;height:19px;"></div></div><div class="glyph-margin-widgets" style="position: absolute; top: 0px;"></div></div><div class="monaco-scrollable-element editor-scrollable vs-dark" role="presentation" data-mprt="5" style="position: absolute; overflow: hidden; left: 6px; width: 1168px; height: 599px;"><div class="lines-content monaco-editor-background" style="position: absolute; overflow: hidden; width: 1e+06px; height: 599px; contain: strict; will-change: unset; top: 0px; left: 0px;"><div class="view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; height: 0px; width: 1168px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line" style="width:1168px; height:19px;"></div></div><div style="position:absolute;top:19px;width:100%;height:19px;"></div><div style="position:absolute;top:38px;width:100%;height:19px;"></div><div style="position:absolute;top:57px;width:100%;height:19px;"></div><div style="position:absolute;top:76px;width:100%;height:19px;"></div><div style="position:absolute;top:95px;width:100%;height:19px;"></div><div style="position:absolute;top:114px;width:100%;height:19px;"></div><div style="position:absolute;top:133px;width:100%;height:19px;"></div><div style="position:absolute;top:152px;width:100%;height:19px;"></div><div style="position:absolute;top:171px;width:100%;height:19px;"></div><div style="position:absolute;top:190px;width:100%;height:19px;"></div><div style="position:absolute;top:209px;width:100%;height:19px;"></div><div style="position:absolute;top:228px;width:100%;height:19px;"></div><div style="position:absolute;top:247px;width:100%;height:19px;"></div><div style="position:absolute;top:266px;width:100%;height:19px;"></div><div style="position:absolute;top:285px;width:100%;height:19px;"></div><div style="position:absolute;top:304px;width:100%;height:19px;"></div><div style="position:absolute;top:323px;width:100%;height:19px;"></div><div style="position:absolute;top:342px;width:100%;height:19px;"></div><div style="position:absolute;top:361px;width:100%;height:19px;"></div><div style="position:absolute;top:380px;width:100%;height:19px;"></div><div style="position:absolute;top:399px;width:100%;height:19px;"></div><div style="position:absolute;top:418px;width:100%;height:19px;"></div><div style="position:absolute;top:437px;width:100%;height:19px;"></div><div style="position:absolute;top:456px;width:100%;height:19px;"></div><div style="position:absolute;top:475px;width:100%;height:19px;"></div><div style="position:absolute;top:494px;width:100%;height:19px;"></div><div style="position:absolute;top:513px;width:100%;height:19px;"></div><div style="position:absolute;top:532px;width:100%;height:19px;"></div><div style="position:absolute;top:551px;width:100%;height:19px;"></div><div style="position:absolute;top:570px;width:100%;height:19px;"></div></div><div role="presentation" aria-hidden="true" class="view-rulers"><div class="view-ruler" style="width: 2px; height: 599px; left: 615.938px;"></div></div><div class="view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="view-lines monaco-mouse-cursor-text" role="presentation" aria-hidden="true" data-mprt="7" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 1168px; height: 599px;"><div style="top:0px;height:19px;" class="view-line"><span><span class="mtk18">import</span><span class="mtk1">&nbsp;pandas&nbsp;</span><span class="mtk18">as</span><span class="mtk1">&nbsp;pd</span></span></div><div style="top:19px;height:19px;" class="view-line"><span><span class="mtk18">import</span><span class="mtk1">&nbsp;matplotlib.pyplot&nbsp;</span><span class="mtk18">as</span><span class="mtk1">&nbsp;plt</span></span></div><div style="top:38px;height:19px;" class="view-line"><span><span class="mtk18">import</span><span class="mtk1">&nbsp;numpy&nbsp;</span><span class="mtk18">as</span><span class="mtk1">&nbsp;np</span></span></div><div style="top:57px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:76px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Load&nbsp;the&nbsp;data</span></span></div><div style="top:95px;height:19px;" class="view-line"><span><span class="mtk1">data&nbsp;=&nbsp;pd.read_csv</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">"diabetes.csv"</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:114px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:133px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Set&nbsp;seed&nbsp;for&nbsp;reproducibility</span></span></div><div style="top:152px;height:19px;" class="view-line"><span><span class="mtk1">np.random.seed</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk6">42</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:171px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:190px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Take&nbsp;a&nbsp;random&nbsp;sample&nbsp;of&nbsp;25&nbsp;observations</span></span></div><div style="top:209px;height:19px;" class="view-line"><span><span class="mtk1">sample&nbsp;=&nbsp;data.sample</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">n=</span><span class="mtk6">25</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:228px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:247px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Population&nbsp;statistics</span></span></div><div style="top:266px;height:19px;" class="view-line"><span><span class="mtk1">pop_mean_glucose&nbsp;=&nbsp;data</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'Glucose'</span><span class="mtk12 bracket-highlighting-0">]</span><span class="mtk1">.mean</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:285px;height:19px;" class="view-line"><span><span class="mtk1">pop_max_glucose&nbsp;=&nbsp;data</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'Glucose'</span><span class="mtk12 bracket-highlighting-0">]</span><span class="mtk1">.</span><span class="mtk15">max</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:304px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:323px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Sample&nbsp;statistics</span></span></div><div style="top:342px;height:19px;" class="view-line"><span><span class="mtk1">sample_mean_glucose&nbsp;=&nbsp;sample</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'Glucose'</span><span class="mtk12 bracket-highlighting-0">]</span><span class="mtk1">.mean</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:361px;height:19px;" class="view-line"><span><span class="mtk1">sample_max_glucose&nbsp;=&nbsp;sample</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'Glucose'</span><span class="mtk12 bracket-highlighting-0">]</span><span class="mtk1">.</span><span class="mtk15">max</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:380px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:399px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Create&nbsp;a&nbsp;bar&nbsp;chart&nbsp;for&nbsp;comparison</span></span></div><div style="top:418px;height:19px;" class="view-line"><span><span class="mtk1">variables&nbsp;=&nbsp;</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'Population&nbsp;Mean'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'Population&nbsp;Max'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'Sample&nbsp;Mean'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'Sample&nbsp;Max'</span><span class="mtk12 bracket-highlighting-0">]</span></span></div><div style="top:437px;height:19px;" class="view-line"><span><span class="mtk1">glucose_values&nbsp;=&nbsp;</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk1">pop_mean_glucose</span><span class="mtk12">,</span><span class="mtk1">&nbsp;pop_max_glucose</span><span class="mtk12">,</span><span class="mtk1">&nbsp;sample_mean_glucose</span><span class="mtk12">,</span><span class="mtk1">&nbsp;sample_max_glucose</span><span class="mtk12 bracket-highlighting-0">]</span></span></div><div style="top:456px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:475px;height:19px;" class="view-line"><span><span class="mtk1">plt.bar</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">variables</span><span class="mtk12">,</span><span class="mtk1">&nbsp;glucose_values</span><span class="mtk12">,</span><span class="mtk1">&nbsp;color=</span><span class="mtk12 bracket-highlighting-1">[</span><span class="mtk5">'blue'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'blue'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'orange'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'orange'</span><span class="mtk12 bracket-highlighting-1">]</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:494px;height:19px;" class="view-line"><span><span class="mtk1">plt.xlabel</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Statistic'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:513px;height:19px;" class="view-line"><span><span class="mtk1">plt.ylabel</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Glucose&nbsp;Value'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:532px;height:19px;" class="view-line"><span><span class="mtk1">plt.title</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Comparison&nbsp;of&nbsp;Glucose&nbsp;Statistics&nbsp;between&nbsp;Populati</span><span class="mtk5">on&nbsp;and&nbsp;Sample'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:551px;height:19px;" class="view-line"><span><span class="mtk1">plt.show</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:570px;height:19px;" class="view-line"><span><span></span></span></div></div><div data-mprt="1" class="contentWidgets" style="position: absolute; top: 0px;"></div><div role="presentation" aria-hidden="true" class="cursors-layer cursor-line-style cursor-solid"><div class="cursor monaco-mouse-cursor-text " style="height: 19px; top: 0px; left: 0px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; display: block; visibility: hidden; padding-left: 0px; width: 1.66667px;"></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute; width: 1154px; height: 10px; left: 0px; bottom: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; width: 1154px;"></div></div><canvas class="decorationsOverviewRuler" aria-hidden="true" width="16" height="718" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; top: 0px; right: 0px; width: 14px; height: 599px; will-change: unset; display: block;"></canvas><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute; width: 14px; height: 599px; right: 0px; top: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 14px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; height: 599px;"></div></div></div><div role="presentation" aria-hidden="true" style="width: 1174px;"></div><textarea data-mprt="6" class="inputarea monaco-mouse-cursor-text" wrap="on" autocorrect="off" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Editor content;Press Alt+F1 for Accessibility Options." tabindex="0" role="textbox" aria-roledescription="editor" aria-multiline="true" aria-haspopup="false" aria-autocomplete="both" style="tab-size: 15.3984px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; top: 0px; left: 6px; width: 1174px; height: 1px;"></textarea><div class="monaco-editor-background textAreaCover" style="position: absolute; top: 0px; left: 0px; width: 0px; height: 0px;"></div><div data-mprt="4" class="overlayWidgets" style="width: 1174px;"><div widgetid="editor.contrib.quickInputWidget" style="position: absolute; top: 0px; right: 50%;"></div></div><div data-mprt="8" class="minimap slider-mouseover" role="presentation" aria-hidden="true" style="position: absolute; left: 0px; width: 0px; height: 599px;"><div class="minimap-shadow-hidden" style="height: 599px;"></div><canvas width="0" height="718" style="position: absolute; left: 0px; width: 0px; height: 599px;"></canvas><canvas class="minimap-decorations-layer" width="0" height="718" style="position: absolute; left: 0px; width: 0px; height: 599px;"></canvas><div class="minimap-slider" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; width: 0px; will-change: unset;"><div class="minimap-slider-horizontal" style="position: absolute; width: 0px; height: 0px;"></div></div></div><div role="presentation" aria-hidden="true" class="blockDecorations-container"></div></div><div data-mprt="2" class="overflowingContentWidgets"><div widgetid="editor.contrib.resizableContentHoverWidget" style="position: fixed; height: 10px; width: 10px; z-index: 50; display: none; visibility: hidden; max-width: 1600px;"><div class="monaco-sash vertical" style="left: 8px;"></div><div class="monaco-sash vertical" style="left: -2px;"></div><div class="monaco-sash orthogonal-edge-north horizontal" style="top: -2px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-sash orthogonal-edge-south horizontal" style="top: 8px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-hover hidden" tabindex="0" role="tooltip"><div class="monaco-scrollable-element " role="presentation" style="position: relative; overflow: hidden;"><div class="monaco-hover-content" style="overflow: hidden; font-size: 14px; line-height: 1.35714; max-width: 774.84px; max-height: 250px;"></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div class="shadow"></div><div class="shadow"></div><div class="shadow"></div></div></div></div></div><div class=".in-cell-overflowing"></div></div></div></div><colab-form class="formview vertical layout flex"><div class="widget-area vertical layout"></div></colab-form></div>
    <div class="output"><!----> <div class="output-header"> </div>
        <div class="output-content">
          <div class="output-info"><colab-output-info title="Clear output

executed at 11:10 PM (0 minutes ago)
executed in 1.41s" hovering=""><template shadowrootmode="open"><!----> <md-icon class="collaborator account-circle" alt="Execution output" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->account_circle</md-icon>
      <!--?lit$2926288814$--> <mwc-icon-button icon="cancel" command="clear-focused-or-selected-outputs" alt="Clear output"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label="cancel"><!--?lit$2926288814$-->
    <!--?lit$2926288814$--><i class="material-icons"><!--?lit$2926288814$-->cancel</i>
    <span><slot></slot></span>
  </button></template>
          </mwc-icon-button><!--?--></template></colab-output-info></div>
          <div class="output-iframe-container">
            <div class="output-iframe-sizer" style="min-height: 0px;"> <div><div class="outputview" style="height: 472px;"><colab-dom-lifecycle-events style="display: none;"></colab-dom-lifecycle-events><iframe allow="accelerometer; autoplay; camera; gyroscope; magnetometer; microphone; serial; usb; xr-spatial-tracking; clipboard-write" sandbox="allow-downloads allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-storage-access-by-user-activation allow-modals allow-popups-to-escape-sandbox" src="./untitled10_files/outputframe.html" class="" style="height: 472px;"></iframe></div></div><div><div></div></div></div>
          </div>
        </div></div></div><colab-cell-next-steps><template shadowrootmode="open"><!----></template></colab-cell-next-steps></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell text" id="cell-FndME8aQcoa6" role="region" aria-label="Cell 3: Text cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><!----> <div class="toolbar-root"><div class="markdown-toolbar"><!----><!--?lit$2926288814$--><!----><mwc-icon-button class="markdown-toolbar-header" title="Toggle heading"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_size</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-bold" title="Bold"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_bold</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-italic" title="Italicize"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_italic</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-code" title="Format as code"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->code</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-link" title="Insert link"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->link</md-icon>
  </mwc-icon-button><!----><!----><label for="markdown-image-input" class="colab-icon markdown-insert-image"><span role="button" title="Insert image" aria-label="Insert image" tabindex="0"><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>image</md-icon></span></label><input id="markdown-image-input" type="file" multiple="" accept="image/*"><!----><!----><mwc-icon-button class="markdown-toolbar-blockquote" title="Add blockquote"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_quote</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-ol" title="Add numbered list"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_numbered</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-ul" title="Add bulleted list"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
            <mwc-ripple unbounded=""><template shadowrootmode="open"><!---->
        <div class="mdc-ripple-surface mdc-ripple-upgraded mdc-ripple-upgraded--unbounded" style="--mdc-ripple-fg-scale: 1.6666666666666667; --mdc-ripple-fg-size: 24px; --mdc-ripple-left: 8px; --mdc-ripple-top: 8px;"></div></template>
            </mwc-ripple>
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_bulleted</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-hr" title="Add horizontal rule"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->horizontal_rule</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="latex" title="LaTeX"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><span>ψ</span>
  </mwc-icon-button><!----><!----><mwc-icon-button title="Insert emoji"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->mood</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-preview" title="Reposition markdown preview"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><g id="markdown-preview-below">
  <rect width="20" height="18" x="2" y="2" rx="2" ry="2" style="fill:none;stroke:var(--colab-icon-color)"></rect>
  <line x1="4.5" y1="13" x2="19.5" y2="13" style="stroke:var(--colab-primary-text-color);stroke-dasharray:2"></line>
  <line x1="2.5" y1="4" x2="21.5" y2="4" style="stroke:var(--colab-icon-color);stroke-width:3px;"></line>
</g></svg></md-icon>
  </mwc-icon-button><!----><!--?--></div></div>
      <div class="editor-container horizontal">
        <div class="editor-root"><div class="editor flex monaco" data-keybinding-context="19" data-mode-id="markdown" style="height: 29px; --vscode-editorCodeLens-lineHeight: 16px; --vscode-editorCodeLens-fontSize: 12px; --vscode-editorCodeLens-fontFeatureSettings: &quot;liga&quot; off, &quot;calt&quot; off; display: none;"><div class="monaco-editor no-user-select  showUnused showDeprecated vs-dark" role="code" data-uri="inmemory://model/26" style="width: 611px; height: 29px;"><div data-mprt="3" class="overflow-guard" style="width: 611px; height: 29px;"><div class="margin" role="presentation" aria-hidden="true" style="position: absolute; contain: strict; will-change: unset; top: 0px; height: 29px; width: 6px;"><div class="glyph-margin" style="left: 0px; width: 0px; height: 29px;"></div><div class="margin-view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="margin-view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 6px; height: 29px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line current-line-margin-both" style="width:6px; height:19px;"></div></div></div><div class="glyph-margin-widgets" style="position: absolute; top: 0px;"></div></div><div class="monaco-scrollable-element editor-scrollable vs-dark" role="presentation" data-mprt="5" style="position: absolute; overflow: hidden; left: 6px; width: 605px; height: 29px;"><div class="lines-content monaco-editor-background" style="position: absolute; overflow: hidden; width: 1e+06px; height: 29px; contain: strict; will-change: unset; top: 0px; left: 0px;"><div class="view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; height: 0px; width: 605px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line" style="width:605px; height:19px;"></div></div></div><div role="presentation" aria-hidden="true" class="view-rulers"></div><div class="view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="view-lines monaco-mouse-cursor-text" role="presentation" aria-hidden="true" data-mprt="7" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 605px; height: 29px;"><div style="top:0px;height:19px;" class="view-line"><span><span class="mtk1">b</span><span class="mtk1 unexpected-closing-bracket">)</span></span></div></div><div data-mprt="1" class="contentWidgets" style="position: absolute; top: 0px;"><div class="lightBulbWidget codicon codicon-light-bulb" widgetid="LightBulbWidget" title="Show Code Actions (Ctrl+.)" style="position: absolute; display: none; visibility: hidden; max-width: 605px;"></div></div><div role="presentation" aria-hidden="true" class="cursors-layer cursor-line-style cursor-solid"><div class="cursor monaco-mouse-cursor-text " style="height: 19px; top: 0px; left: 15px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; display: block; visibility: hidden; padding-left: 0px; width: 1.66667px;"></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute; width: 591px; height: 10px; left: 0px; bottom: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; width: 591px;"></div></div><canvas class="decorationsOverviewRuler" aria-hidden="true" width="16" height="34" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; top: 0px; right: 0px; width: 14px; height: 29px; will-change: unset; display: block;"></canvas><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute; width: 14px; height: 29px; right: 0px; top: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 14px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; height: 29px;"></div></div></div><div role="presentation" aria-hidden="true" style="width: 611px;"></div><textarea data-mprt="6" class="inputarea monaco-mouse-cursor-text" wrap="on" autocorrect="off" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Editor content;Press Alt+F1 for Accessibility Options." tabindex="0" role="textbox" aria-roledescription="editor" aria-multiline="true" aria-haspopup="false" aria-autocomplete="both" style="tab-size: 15.3984px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; top: 0px; left: 6px; width: 585px; height: 1px;"></textarea><div style="position: absolute; top: 0px; left: 0px; width: 0px; height: 0px;" class="monaco-editor-background textAreaCover"></div><div data-mprt="4" class="overlayWidgets" style="width: 611px;"></div><div data-mprt="8" class="minimap slider-mouseover" role="presentation" aria-hidden="true" style="position: absolute; left: 0px; width: 0px; height: 29px;"><div class="minimap-shadow-hidden" style="height: 29px;"></div><canvas width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><canvas class="minimap-decorations-layer" width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><div class="minimap-slider" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; width: 0px; will-change: unset;"><div class="minimap-slider-horizontal" style="position: absolute; width: 0px; height: 0px;"></div></div></div><div role="presentation" aria-hidden="true" class="blockDecorations-container"></div></div><div data-mprt="2" class="overflowingContentWidgets" style="display: none;"><div widgetid="editor.contrib.resizableContentHoverWidget" style="position: fixed; height: 10px; width: 10px; z-index: 50; display: none; visibility: hidden; max-width: 1600px;"><div class="monaco-sash vertical" style="left: 8px;"></div><div class="monaco-sash vertical" style="left: -2px;"></div><div class="monaco-sash orthogonal-edge-north horizontal" style="top: -2px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-sash orthogonal-edge-south horizontal" style="top: 8px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-hover hidden" role="tooltip" tabindex="0"><div class="monaco-scrollable-element " role="presentation" style="position: relative; overflow: hidden;"><div class="monaco-hover-content" style="overflow: hidden; font-size: 14px; line-height: 1.35714; max-width: 500px; max-height: 250px;"></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div class="shadow"></div><div class="shadow"></div><div class="shadow"></div></div></div></div></div><div class=".in-cell-overflowing"><div widgetid="editor.contrib.quickInputWidget" style="position: absolute; top: 0px; right: 50%;"></div></div></div></div></div>
        <div class="text-top-div">
          <div class="markdown"><span><p>b)</p>
</span></div>
        </div>
      </div>

      <div class="section-header" style="display: none;">
        <colab-run-button role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution">
      <div class="execution-count"><!--?lit$2926288814$-->[ ]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><!--?-->
    </div></template></colab-run-button>
        <div class="section-header-container" title="click to expand">↳ 0 cells hidden</div>
      </div></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell code icon-scrolling" id="cell-lEc6yatrcsBp" role="region" aria-label="Cell 4: Code cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><div class="codecell-input-output">
      <div class="inputarea horizontal layout code">
        <div class="cell-gutter">
          <!-- Bounding range for vertical scrolling of icons -->
          <div class="cell-execution-container">
            <colab-run-button title="Run cell (Ctrl+Enter)
cell has not been executed in this session" role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution stale hovered">
      <div class="execution-count"><!--?lit$2926288814$-->[ ]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><!--?-->
    </div></template></colab-run-button>
          </div>
        </div>
      <div class="editor flex lazy-editor" style=""><pre class="lazy-virtualized" style="font-size: 14px; line-height: 19px;"><pre class="lazy-gutter"></pre><pre class="monaco-colorized colab-dark colab-dark colab-dark colab-dark colab-dark colab-dark colab-dark colab-dark colab-dark" data-lang="notebook-python"><span><span></span></span><br></pre><colab-read-only-cell-placeholder style="display: block;"><template shadowrootmode="open"><!----><div><!--?lit$2926288814$-->Start coding or <span role="button" class="link" tabindex="0">generate</span> with AI.</div></template></colab-read-only-cell-placeholder></pre></div><colab-form class="formview vertical layout flex"><div class="widget-area vertical layout"></div></colab-form></div>
    <div class="output"><!----> <div class="output-header"> </div>
        <div class="output-content" style="display: none;">
          <div class="output-info"> </div>
          <div class="output-iframe-container">
            <div class="output-iframe-sizer"> <div><div></div></div></div>
          </div>
        </div></div></div><colab-cell-next-steps><template shadowrootmode="open"><!----></template></colab-cell-next-steps></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell code icon-scrolling code-has-output" id="cell-cXUT8wNpsji1" role="region" aria-label="Cell 5: Code cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><div class="codecell-input-output">
      <div class="inputarea horizontal layout code">
        <div class="cell-gutter">
          <!-- Bounding range for vertical scrolling of icons -->
          <div class="cell-execution-container">
            <colab-run-button title="Run cell (Ctrl+Enter)
cell executed since last change

executed at 11:10 PM (2 minutes ago)
executed in 0.283s" role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution">
      <div class="execution-count"><!--?lit$2926288814$-->[6]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><div id="status">
      <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></svg></md-icon>
      <div><!--?lit$2926288814$-->0s</div>
    </div>
    </div></template></colab-run-button>
          </div>
        </div>
      <div class="editor flex lazy-editor" style=""><pre class="lazy-virtualized" style="font-size: 14px; line-height: 19px;"><pre class="lazy-gutter"></pre><pre class="monaco-colorized colab-dark colab-dark colab-dark colab-dark colab-dark" data-lang="notebook-python"><span><span class="mtk8">#&nbsp;Population&nbsp;percentile</span></span><br><span><span class="mtk1">pop_98th_percentile_bmi&nbsp;=&nbsp;np.percentile</span><span class="mtk12">(</span><span class="mtk1">data</span><span class="mtk12">[</span><span class="mtk5">'BMI'</span><span class="mtk12">],</span><span class="mtk1">&nbsp;</span><span class="mtk6">98</span><span class="mtk12">)</span></span><br><span><span></span></span><br><span><span class="mtk8">#&nbsp;Sample&nbsp;percentile</span></span><br><span><span class="mtk1">sample_98th_percentile_bmi&nbsp;=&nbsp;np.percentile</span><span class="mtk12">(</span><span class="mtk1">sample</span><span class="mtk12">[</span><span class="mtk5">'BMI'</span><span class="mtk12">],</span><span class="mtk1">&nbsp;</span><span class="mtk6">98</span><span class="mtk12">)</span></span><br><span><span></span></span><br><span><span class="mtk8">#&nbsp;Create&nbsp;a&nbsp;bar&nbsp;chart&nbsp;for&nbsp;comparison</span></span><br><span><span class="mtk1">variables&nbsp;=&nbsp;</span><span class="mtk12">[</span><span class="mtk5">'Population'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'Sample'</span><span class="mtk12">]</span></span><br><span><span class="mtk1">bmi_values&nbsp;=&nbsp;</span><span class="mtk12">[</span><span class="mtk1">pop_98th_percentile_bmi</span><span class="mtk12">,</span><span class="mtk1">&nbsp;sample_98th_percentile_bmi</span><span class="mtk12">]</span></span><br><span><span></span></span><br><span><span class="mtk1">plt.bar</span><span class="mtk12">(</span><span class="mtk1">variables</span><span class="mtk12">,</span><span class="mtk1">&nbsp;bmi_values</span><span class="mtk12">,</span><span class="mtk1">&nbsp;color=</span><span class="mtk12">[</span><span class="mtk5">'blue'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk5">'orange'</span><span class="mtk12">])</span></span><br><span><span class="mtk1">plt.xlabel</span><span class="mtk12">(</span><span class="mtk5">'Data'</span><span class="mtk12">)</span></span><br><span><span class="mtk1">plt.ylabel</span><span class="mtk12">(</span><span class="mtk5">'98th&nbsp;Percentile&nbsp;of&nbsp;BMI'</span><span class="mtk12">)</span></span><br><span><span class="mtk1">plt.title</span><span class="mtk12">(</span><span class="mtk5">'Comparison&nbsp;of&nbsp;98th&nbsp;Percentile&nbsp;of&nbsp;BMI&nbsp;between&nbsp;Popu</span><span class="mtk5">lation&nbsp;and&nbsp;Sample'</span><span class="mtk12">)</span></span><br><span><span class="mtk1">plt.show</span><span class="mtk12">()</span></span><br><span><span></span></span><br></pre><colab-read-only-cell-placeholder style="display: none;"><template shadowrootmode="open"><!----><div><!--?lit$2926288814$-->Start coding or <span role="button" class="link" tabindex="0">generate</span> with AI.</div></template></colab-read-only-cell-placeholder></pre></div><colab-form class="formview vertical layout flex"><div class="widget-area vertical layout"></div></colab-form></div>
    <div class="output"><!----> <div class="output-header"> </div>
        <div class="output-content">
          <div class="output-info"><colab-output-info title="Clear output

executed at 11:10 PM (0 minutes ago)
executed in 0.283s" hovering=""><template shadowrootmode="open"><!----> <md-icon class="collaborator account-circle" alt="Execution output" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->account_circle</md-icon>
      <!--?lit$2926288814$--> <mwc-icon-button icon="cancel" command="clear-focused-or-selected-outputs" alt="Clear output"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label="cancel"><!--?lit$2926288814$-->
    <!--?lit$2926288814$--><i class="material-icons"><!--?lit$2926288814$-->cancel</i>
    <span><slot></slot></span>
  </button></template>
          </mwc-icon-button><!--?--></template></colab-output-info></div>
          <div class="output-iframe-container">
            <div class="output-iframe-sizer" style="min-height: 0px;"> <div><div class="outputview" style="height: 472px;"><colab-dom-lifecycle-events style="display: none;"></colab-dom-lifecycle-events><iframe allow="accelerometer; autoplay; camera; gyroscope; magnetometer; microphone; serial; usb; xr-spatial-tracking; clipboard-write" sandbox="allow-downloads allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-storage-access-by-user-activation allow-modals allow-popups-to-escape-sandbox" src="./untitled10_files/outputframe(1).html" class="" style="height: 472px;"></iframe></div></div></div>
          </div>
        </div></div></div><colab-cell-next-steps><template shadowrootmode="open"><!----></template></colab-cell-next-steps></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell text" id="cell-PLaSWZsncwDa" role="region" aria-label="Cell 6: Text cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><!----> <div class="toolbar-root"><div class="markdown-toolbar"><!----><!--?lit$2926288814$--><!----><mwc-icon-button class="markdown-toolbar-header" title="Toggle heading"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_size</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-bold" title="Bold"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_bold</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-italic" title="Italicize"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_italic</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-code" title="Format as code"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->code</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-link" title="Insert link"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->link</md-icon>
  </mwc-icon-button><!----><!----><label for="markdown-image-input" class="colab-icon markdown-insert-image"><span role="button" title="Insert image" aria-label="Insert image" tabindex="0"><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>image</md-icon></span></label><input id="markdown-image-input" type="file" multiple="" accept="image/*"><!----><!----><mwc-icon-button class="markdown-toolbar-blockquote" title="Add blockquote"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_quote</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-ol" title="Add numbered list"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_numbered</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-ul" title="Add bulleted list"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->format_list_bulleted</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-hr" title="Add horizontal rule"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->horizontal_rule</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="latex" title="LaTeX"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><span>ψ</span>
  </mwc-icon-button><!----><!----><mwc-icon-button title="Insert emoji"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->mood</md-icon>
  </mwc-icon-button><!----><!----><mwc-icon-button class="markdown-toolbar-preview" title="Reposition markdown preview"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label=""><!--?lit$2926288814$-->
    <!--?lit$2926288814$-->
    <span><slot></slot></span>
  </button></template>
    <!--?lit$2926288814$--><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><g id="markdown-preview-below">
  <rect width="20" height="18" x="2" y="2" rx="2" ry="2" style="fill:none;stroke:var(--colab-icon-color)"></rect>
  <line x1="4.5" y1="13" x2="19.5" y2="13" style="stroke:var(--colab-primary-text-color);stroke-dasharray:2"></line>
  <line x1="2.5" y1="4" x2="21.5" y2="4" style="stroke:var(--colab-icon-color);stroke-width:3px;"></line>
</g></svg></md-icon>
  </mwc-icon-button><!----><!--?--></div></div>
      <div class="editor-container horizontal">
        <div class="editor-root"><div class="editor flex monaco" data-keybinding-context="25" data-mode-id="markdown" style="height: 29px; --vscode-editorCodeLens-lineHeight: 16px; --vscode-editorCodeLens-fontSize: 12px; --vscode-editorCodeLens-fontFeatureSettings: &quot;liga&quot; off, &quot;calt&quot; off; display: none;"><div class="monaco-editor no-user-select  showUnused showDeprecated vs-dark" role="code" data-uri="inmemory://model/29" style="width: 611px; height: 29px;"><div data-mprt="3" class="overflow-guard" style="width: 611px; height: 29px;"><div class="margin" role="presentation" aria-hidden="true" style="position: absolute; contain: strict; will-change: unset; top: 0px; height: 29px; width: 6px;"><div class="glyph-margin" style="left: 0px; width: 0px; height: 29px;"></div><div class="margin-view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="margin-view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 6px; height: 29px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line current-line-margin-both" style="width:6px; height:19px;"></div></div></div><div class="glyph-margin-widgets" style="position: absolute; top: 0px;"></div></div><div class="monaco-scrollable-element editor-scrollable vs-dark" role="presentation" data-mprt="5" style="position: absolute; overflow: hidden; left: 6px; width: 605px; height: 29px;"><div class="lines-content monaco-editor-background" style="position: absolute; overflow: hidden; width: 1e+06px; height: 29px; contain: strict; will-change: unset; top: 0px; left: 0px;"><div class="view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; height: 0px; width: 605px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line" style="width:605px; height:19px;"></div></div></div><div role="presentation" aria-hidden="true" class="view-rulers"></div><div class="view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="view-lines monaco-mouse-cursor-text" role="presentation" aria-hidden="true" data-mprt="7" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 605px; height: 29px;"><div style="top:0px;height:19px;" class="view-line"><span><span class="mtk1">c</span><span class="mtk1 unexpected-closing-bracket">)</span></span></div></div><div data-mprt="1" class="contentWidgets" style="position: absolute; top: 0px;"><div class="lightBulbWidget codicon codicon-light-bulb" widgetid="LightBulbWidget" title="Show Code Actions (Ctrl+.)" style="position: absolute; display: none; visibility: hidden; max-width: 605px;"></div></div><div role="presentation" aria-hidden="true" class="cursors-layer cursor-line-style cursor-solid"><div class="cursor monaco-mouse-cursor-text " style="height: 19px; top: 0px; left: 15px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; display: block; visibility: hidden; padding-left: 0px; width: 1.66667px;"></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute; width: 591px; height: 10px; left: 0px; bottom: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; width: 591px;"></div></div><canvas class="decorationsOverviewRuler" aria-hidden="true" width="16" height="34" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; top: 0px; right: 0px; width: 14px; height: 29px; will-change: unset; display: block;"></canvas><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute; width: 14px; height: 29px; right: 0px; top: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 14px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; height: 29px;"></div></div></div><div role="presentation" aria-hidden="true" style="width: 611px;"></div><textarea data-mprt="6" class="inputarea monaco-mouse-cursor-text" wrap="on" autocorrect="off" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Editor content;Press Alt+F1 for Accessibility Options." tabindex="0" role="textbox" aria-roledescription="editor" aria-multiline="true" aria-haspopup="false" aria-autocomplete="both" style="tab-size: 15.3984px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; top: 0px; left: 6px; width: 585px; height: 1px;"></textarea><div style="position: absolute; top: 0px; left: 0px; width: 0px; height: 0px;" class="monaco-editor-background textAreaCover"></div><div data-mprt="4" class="overlayWidgets" style="width: 611px;"></div><div data-mprt="8" class="minimap slider-mouseover" role="presentation" aria-hidden="true" style="position: absolute; left: 0px; width: 0px; height: 29px;"><div class="minimap-shadow-hidden" style="height: 29px;"></div><canvas width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><canvas class="minimap-decorations-layer" width="0" height="34" style="position: absolute; left: 0px; width: 0px; height: 29px;"></canvas><div class="minimap-slider" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; width: 0px; will-change: unset;"><div class="minimap-slider-horizontal" style="position: absolute; width: 0px; height: 0px;"></div></div></div><div role="presentation" aria-hidden="true" class="blockDecorations-container"></div></div><div data-mprt="2" class="overflowingContentWidgets" style="display: none;"><div widgetid="editor.contrib.resizableContentHoverWidget" style="position: fixed; height: 10px; width: 10px; z-index: 50; display: none; visibility: hidden; max-width: 1600px;"><div class="monaco-sash vertical" style="left: 8px;"></div><div class="monaco-sash vertical" style="left: -2px;"></div><div class="monaco-sash orthogonal-edge-north horizontal" style="top: -2px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-sash orthogonal-edge-south horizontal" style="top: 8px;"><div class="orthogonal-drag-handle start"></div><div class="orthogonal-drag-handle end"></div></div><div class="monaco-hover hidden" role="tooltip" tabindex="0"><div class="monaco-scrollable-element " role="presentation" style="position: relative; overflow: hidden;"><div class="monaco-hover-content" style="overflow: hidden; font-size: 14px; line-height: 1.35714; max-width: 500px; max-height: 250px;"></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div class="shadow"></div><div class="shadow"></div><div class="shadow"></div></div></div></div></div><div class=".in-cell-overflowing"><div widgetid="editor.contrib.quickInputWidget" style="position: absolute; top: 0px; right: 50%;"></div></div></div></div></div>
        <div class="text-top-div">
          <div class="markdown"><span><p>c)</p>
</span></div>
        </div>
      </div>

      <div class="section-header" style="display: none;">
        <colab-run-button role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution">
      <div class="execution-count"><!--?lit$2926288814$-->[ ]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><!--?-->
    </div></template></colab-run-button>
        <div class="section-header-container" title="click to expand">↳ 0 cells hidden</div>
      </div></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface"></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell code icon-scrolling" id="cell-vnZmbC-Kcwzj" role="region" aria-label="Cell 7: Code cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"></div><div class="main-content" elevation="0"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><div class="codecell-input-output">
      <div class="inputarea horizontal layout code">
        <div class="cell-gutter">
          <!-- Bounding range for vertical scrolling of icons -->
          <div class="cell-execution-container">
            <colab-run-button title="Run cell (Ctrl+Enter)
cell has not been executed in this session" role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution stale">
      <div class="execution-count"><!--?lit$2926288814$-->[ ]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><!--?-->
    </div></template></colab-run-button>
          </div>
        </div>
      <div class="editor flex lazy-editor" style=""><pre class="lazy-virtualized" style="font-size: 14px; line-height: 19px;"><pre class="lazy-gutter"></pre><pre class="monaco-colorized colab-dark colab-dark colab-dark colab-dark colab-dark" data-lang="notebook-python"><span><span></span></span><br></pre><colab-read-only-cell-placeholder style="display: block;"><template shadowrootmode="open"><!----><div><!--?lit$2926288814$-->Start coding or <span role="button" class="link" tabindex="0">generate</span> with AI.</div></template></colab-read-only-cell-placeholder></pre></div><colab-form class="formview vertical layout flex"><div class="widget-area vertical layout"></div></colab-form></div>
    <div class="output"><!----> <div class="output-header"> </div>
        <div class="output-content" style="display: none;">
          <div class="output-info"> </div>
          <div class="output-iframe-container">
            <div class="output-iframe-sizer"> <div><div></div></div></div>
          </div>
        </div></div></div><colab-cell-next-steps><template shadowrootmode="open"><!----></template></colab-cell-next-steps></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div><div class="cell code icon-scrolling code-has-output focused" id="cell-r69rv1NFa3-w" role="region" aria-label="Cell 8: Code cell: " style="opacity: 1;" tabindex="-1"><div class="cell-tag-editor sticky"></div><div class="cell-toolbar sticky"><colab-cell-toolbar><template shadowrootmode="open"><!----><!--?lit$2926288814$--><!----> <md-icon-button class="colab-icon" title="Move cell up
Ctrl+M K" aria-label="Move cell up
Ctrl+M K" command="move-cell-up" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Move cell up
Ctrl+M K">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->arrow_upward</md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Move cell down
Ctrl+M J" aria-label="Move cell down
Ctrl+M J" command="move-cell-down" role="presentation" disabled="" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Move cell down
Ctrl+M J" disabled="">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple disabled="" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->arrow_downward</md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Copy link to cell" aria-label="Copy link to cell" command="copy-link-to-cell" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Copy link to cell">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->link</md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Add a comment
Ctrl+Alt+M" aria-label="Add a comment
Ctrl+Alt+M" command="add-comment" role="presentation" value="" style="display: none;"><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Add a comment
Ctrl+Alt+M">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->comment</md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Open editor settings" aria-label="Open editor settings" command="editor-preferences" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Open editor settings">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon filled="" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->settings</md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Edit" aria-label="Edit" command="toggle-edit-markdown" toggle="" role="presentation" value="" style="display: none;"><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Edit" aria-pressed="false">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->edit</md-icon>
          <!--?lit$2926288814$--><md-icon slot="selected" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->edit_off</md-icon>
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Mirror cell in tab" aria-label="Mirror cell in tab" command="mirror-cell-in-tab" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Mirror cell in tab">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$-->
      <g id="mirror-cell">
        <path d="M4,21V7H2V21a2,2,0,0,0,2,2H18V21Z"></path>
        <path d="M6,13v2H8.6L5,18.6,6.4,20,10,16.4V19h2V13Z"></path>
        <path d="M19,1H8A2,2,0,0,0,6,3v8H8V3H19V17H14v2h5a2,2,0,0,0,2-2V3A2,2,0,0,0,19,1Z"></path>
      </g></svg></md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!----> <md-icon-button class="colab-icon" title="Delete cell
Ctrl+M D" aria-label="Delete cell
Ctrl+M D" command="delete-cell-or-selection" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Delete cell
Ctrl+M D">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
          <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->delete</md-icon>
          <!--?lit$2926288814$-->
        </md-icon-button><!----><!--?lit$2926288814$--><md-icon-button title="More cell actions" aria-label="More cell actions" class="colab-icon cell-toolbar-more" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="More cell actions">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_vert</md-icon>
    </md-icon-button><!--?--></template></colab-cell-toolbar></div><div class="main-content" elevation="2"><div class="cell-contents"><div class="cell-mask"></div><span class="imported-info-area"></span><div class="codecell-input-output">
      <div class="inputarea horizontal layout code">
        <div class="cell-gutter">
          <!-- Bounding range for vertical scrolling of icons -->
          <div class="cell-execution-container">
            <colab-run-button title="Run cell (Ctrl+Enter)
cell executed since last change

executed at 11:11 PM (1 minute ago)
executed in 1.065s" role="button" aria-label="Run cell"><template shadowrootmode="open"><!----> <div class="cell-execution focused">
      <div class="execution-count"><!--?lit$2926288814$-->[7]</div>
      <div class="cell-execution-indicator"> <!--?lit$2926288814$-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <!--?lit$2926288814$-->
  <mask id="playSymbolMask">
    <rect width="100%" height="100%" fill="white"></rect>
    <polygon points="10,8 17,12 10,16" fill="black"></polygon>
  </mask>
  <circle cx="12" cy="12" r="7.8" mask="url(#playSymbolMask)" id="filledCircle"></circle>
</svg> </div>
      <!--?lit$2926288814$--><div id="status">
      <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></svg></md-icon>
      <div><!--?lit$2926288814$-->1s</div>
    </div>
    </div></template></colab-run-button>
          </div>
        </div>
      <div class="editor flex lazy-editor" style=""><div class="editor flex monaco" data-keybinding-context="27" data-mode-id="notebook-python" style="height: 998px; --vscode-editorCodeLens-lineHeight: 16px; --vscode-editorCodeLens-fontSize: 12px; --vscode-editorCodeLens-fontFeatureSettings: &quot;liga&quot; off, &quot;calt&quot; off;"><div class="monaco-editor no-user-select  showUnused showDeprecated vs-dark" role="code" data-uri="inmemory://model/31" style="width: 1174px; height: 998px;"><div data-mprt="3" class="overflow-guard" style="width: 1174px; height: 998px;"><div class="margin" role="presentation" aria-hidden="true" style="position: absolute; contain: strict; will-change: unset; top: 0px; height: 998px; width: 6px;"><div class="glyph-margin" style="left: 0px; width: 0px; height: 998px;"></div><div class="margin-view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="margin-view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 6px; height: 998px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line current-line-margin-both" style="width:6px; height:19px;"></div></div><div style="position:absolute;top:19px;width:100%;height:19px;"></div><div style="position:absolute;top:38px;width:100%;height:19px;"></div><div style="position:absolute;top:57px;width:100%;height:19px;"></div><div style="position:absolute;top:76px;width:100%;height:19px;"></div><div style="position:absolute;top:95px;width:100%;height:19px;"></div><div style="position:absolute;top:114px;width:100%;height:19px;"></div><div style="position:absolute;top:133px;width:100%;height:19px;"></div><div style="position:absolute;top:152px;width:100%;height:19px;"></div><div style="position:absolute;top:171px;width:100%;height:19px;"></div><div style="position:absolute;top:190px;width:100%;height:19px;"></div><div style="position:absolute;top:209px;width:100%;height:19px;"></div><div style="position:absolute;top:228px;width:100%;height:19px;"></div><div style="position:absolute;top:247px;width:100%;height:19px;"></div><div style="position:absolute;top:266px;width:100%;height:19px;"></div><div style="position:absolute;top:285px;width:100%;height:19px;"></div><div style="position:absolute;top:304px;width:100%;height:19px;"></div><div style="position:absolute;top:323px;width:100%;height:19px;"></div><div style="position:absolute;top:342px;width:100%;height:19px;"></div><div style="position:absolute;top:361px;width:100%;height:19px;"></div><div style="position:absolute;top:380px;width:100%;height:19px;"></div><div style="position:absolute;top:399px;width:100%;height:19px;"></div><div style="position:absolute;top:418px;width:100%;height:19px;"></div><div style="position:absolute;top:437px;width:100%;height:19px;"></div><div style="position:absolute;top:456px;width:100%;height:19px;"></div><div style="position:absolute;top:475px;width:100%;height:19px;"></div><div style="position:absolute;top:494px;width:100%;height:19px;"></div><div style="position:absolute;top:513px;width:100%;height:19px;"></div><div style="position:absolute;top:532px;width:100%;height:19px;"></div><div style="position:absolute;top:551px;width:100%;height:19px;"></div><div style="position:absolute;top:570px;width:100%;height:19px;"></div><div style="position:absolute;top:589px;width:100%;height:19px;"></div><div style="position:absolute;top:608px;width:100%;height:19px;"></div><div style="position:absolute;top:627px;width:100%;height:19px;"></div><div style="position:absolute;top:646px;width:100%;height:19px;"></div><div style="position:absolute;top:665px;width:100%;height:19px;"></div><div style="position:absolute;top:684px;width:100%;height:19px;"></div><div style="position:absolute;top:703px;width:100%;height:19px;"></div><div style="position:absolute;top:722px;width:100%;height:19px;"></div><div style="position:absolute;top:741px;width:100%;height:19px;"></div><div style="position:absolute;top:760px;width:100%;height:19px;"></div><div style="position:absolute;top:779px;width:100%;height:19px;"></div><div style="position:absolute;top:798px;width:100%;height:19px;"></div><div style="position:absolute;top:817px;width:100%;height:19px;"></div><div style="position:absolute;top:836px;width:100%;height:19px;"></div><div style="position:absolute;top:855px;width:100%;height:19px;"></div><div style="position:absolute;top:874px;width:100%;height:19px;"></div><div style="position:absolute;top:893px;width:100%;height:19px;"></div><div style="position:absolute;top:912px;width:100%;height:19px;"></div><div style="position:absolute;top:931px;width:100%;height:19px;"></div><div style="position:absolute;top:950px;width:100%;height:19px;"></div><div style="position:absolute;top:969px;width:100%;height:19px;"></div></div><div class="glyph-margin-widgets" style="position: absolute; top: 0px;"></div></div><div class="monaco-scrollable-element editor-scrollable vs-dark" role="presentation" data-mprt="5" style="position: absolute; overflow: hidden; left: 6px; width: 1168px; height: 998px;"><div class="lines-content monaco-editor-background" style="position: absolute; overflow: hidden; width: 1e+06px; height: 998px; contain: strict; will-change: unset; top: 0px; left: 0px;"><div class="view-overlays" role="presentation" aria-hidden="true" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; height: 0px; width: 1168px;"><div style="position:absolute;top:0px;width:100%;height:19px;"><div class="current-line" style="width:1168px; height:19px;"></div></div><div style="position:absolute;top:19px;width:100%;height:19px;"></div><div style="position:absolute;top:38px;width:100%;height:19px;"></div><div style="position:absolute;top:57px;width:100%;height:19px;"></div><div style="position:absolute;top:76px;width:100%;height:19px;"></div><div style="position:absolute;top:95px;width:100%;height:19px;"></div><div style="position:absolute;top:114px;width:100%;height:19px;"></div><div style="position:absolute;top:133px;width:100%;height:19px;"></div><div style="position:absolute;top:152px;width:100%;height:19px;"></div><div style="position:absolute;top:171px;width:100%;height:19px;"></div><div style="position:absolute;top:190px;width:100%;height:19px;"></div><div style="position:absolute;top:209px;width:100%;height:19px;"></div><div style="position:absolute;top:228px;width:100%;height:19px;"></div><div style="position:absolute;top:247px;width:100%;height:19px;"></div><div style="position:absolute;top:266px;width:100%;height:19px;"></div><div style="position:absolute;top:285px;width:100%;height:19px;"></div><div style="position:absolute;top:304px;width:100%;height:19px;"></div><div style="position:absolute;top:323px;width:100%;height:19px;"></div><div style="position:absolute;top:342px;width:100%;height:19px;"></div><div style="position:absolute;top:361px;width:100%;height:19px;"></div><div style="position:absolute;top:380px;width:100%;height:19px;"></div><div style="position:absolute;top:399px;width:100%;height:19px;"></div><div style="position:absolute;top:418px;width:100%;height:19px;"></div><div style="position:absolute;top:437px;width:100%;height:19px;"></div><div style="position:absolute;top:456px;width:100%;height:19px;"></div><div style="position:absolute;top:475px;width:100%;height:19px;"></div><div style="position:absolute;top:494px;width:100%;height:19px;"></div><div style="position:absolute;top:513px;width:100%;height:19px;"></div><div style="position:absolute;top:532px;width:100%;height:19px;"></div><div style="position:absolute;top:551px;width:100%;height:19px;"></div><div style="position:absolute;top:570px;width:100%;height:19px;"></div><div style="position:absolute;top:589px;width:100%;height:19px;"></div><div style="position:absolute;top:608px;width:100%;height:19px;"></div><div style="position:absolute;top:627px;width:100%;height:19px;"></div><div style="position:absolute;top:646px;width:100%;height:19px;"></div><div style="position:absolute;top:665px;width:100%;height:19px;"></div><div style="position:absolute;top:684px;width:100%;height:19px;"></div><div style="position:absolute;top:703px;width:100%;height:19px;"></div><div style="position:absolute;top:722px;width:100%;height:19px;"></div><div style="position:absolute;top:741px;width:100%;height:19px;"></div><div style="position:absolute;top:760px;width:100%;height:19px;"></div><div style="position:absolute;top:779px;width:100%;height:19px;"></div><div style="position:absolute;top:798px;width:100%;height:19px;"></div><div style="position:absolute;top:817px;width:100%;height:19px;"></div><div style="position:absolute;top:836px;width:100%;height:19px;"></div><div style="position:absolute;top:855px;width:100%;height:19px;"></div><div style="position:absolute;top:874px;width:100%;height:19px;"></div><div style="position:absolute;top:893px;width:100%;height:19px;"></div><div style="position:absolute;top:912px;width:100%;height:19px;"></div><div style="position:absolute;top:931px;width:100%;height:19px;"></div><div style="position:absolute;top:950px;width:100%;height:19px;"></div><div style="position:absolute;top:969px;width:100%;height:19px;"></div></div><div role="presentation" aria-hidden="true" class="view-rulers"><div class="view-ruler" style="width: 2px; height: 998px; left: 615.938px;"></div></div><div class="view-zones" role="presentation" aria-hidden="true" style="position: absolute;"></div><div class="view-lines monaco-mouse-cursor-text" role="presentation" aria-hidden="true" data-mprt="7" style="position: absolute; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; width: 1168px; height: 998px;"><div style="top:0px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Define&nbsp;a&nbsp;function&nbsp;for&nbsp;bootstrap&nbsp;sampling</span></span></div><div style="top:19px;height:19px;" class="view-line"><span><span class="mtk9">def</span><span class="mtk1">&nbsp;</span><span class="mtk15">bootstrap_sample</span><span class="mtk1 bracket-highlighting-0">(</span><span class="mtk4">data</span><span class="mtk1">,&nbsp;</span><span class="mtk4">n_samples</span><span class="mtk1">,&nbsp;</span><span class="mtk4">sample_size</span><span class="mtk1 bracket-highlighting-0">)</span><span class="mtk12">:</span></span></div><div style="top:38px;height:19px;" class="view-line"><span><span class="mtk1">&nbsp;&nbsp;&nbsp;&nbsp;samples&nbsp;=&nbsp;</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk12 bracket-highlighting-0">]</span></span></div><div style="top:57px;height:19px;" class="view-line"><span><span class="mtk1">&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="mtk18">for</span><span class="mtk1">&nbsp;_&nbsp;</span><span class="mtk17">in</span><span class="mtk1">&nbsp;</span><span class="mtk15">range</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">n_samples</span><span class="mtk12 bracket-highlighting-0">)</span><span class="mtk12">:</span></span></div><div style="top:76px;height:19px;" class="view-line"><span><span class="mtk1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sample&nbsp;=&nbsp;data.sample</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">n=sample_size</span><span class="mtk12">,</span><span class="mtk1">&nbsp;replace=</span><span class="mtk9">True</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:95px;height:19px;" class="view-line"><span><span class="mtk1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;samples.append</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">sample</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:114px;height:19px;" class="view-line"><span><span class="mtk1">&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="mtk18">return</span><span class="mtk1">&nbsp;samples</span></span></div><div style="top:133px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:152px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Bootstrap&nbsp;parameters</span></span></div><div style="top:171px;height:19px;" class="view-line"><span><span class="mtk1">n_bootstrap_samples&nbsp;=&nbsp;</span><span class="mtk6">500</span></span></div><div style="top:190px;height:19px;" class="view-line"><span><span class="mtk1">bootstrap_sample_size&nbsp;=&nbsp;</span><span class="mtk6">150</span></span></div><div style="top:209px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:228px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Perform&nbsp;bootstrap&nbsp;sampling</span></span></div><div style="top:247px;height:19px;" class="view-line"><span><span class="mtk1">bootstrap_samples&nbsp;=&nbsp;bootstrap_sample</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">data</span><span class="mtk12">,</span><span class="mtk1">&nbsp;n_bootstrap_samples</span><span class="mtk12">,</span><span class="mtk1">&nbsp;bootstrap_sample_size</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:266px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:285px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Calculate&nbsp;statistics&nbsp;for&nbsp;each&nbsp;bootstrap&nbsp;sample</span></span></div><div style="top:304px;height:19px;" class="view-line"><span><span class="mtk1">bootstrap_mean_blood_pressure&nbsp;=&nbsp;</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk1">sample</span><span class="mtk12 bracket-highlighting-1">[</span><span class="mtk5">'BloodPressure'</span><span class="mtk12 bracket-highlighting-1">]</span><span class="mtk1">.mean</span><span class="mtk12 bracket-highlighting-1">(</span><span class="mtk12 bracket-highlighting-1">)</span><span class="mtk1">&nbsp;</span><span class="mtk18">for</span><span class="mtk1">&nbsp;sample&nbsp;</span><span class="mtk17">in</span><span class="mtk1">&nbsp;bootstrap_samples</span><span class="mtk12 bracket-highlighting-0">]</span></span></div><div style="top:323px;height:19px;" class="view-line"><span><span class="mtk1">bootstrap_std_blood_pressure&nbsp;=&nbsp;</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk1">sample</span><span class="mtk12 bracket-highlighting-1">[</span><span class="mtk5">'BloodPressure'</span><span class="mtk12 bracket-highlighting-1">]</span><span class="mtk1">.std</span><span class="mtk12 bracket-highlighting-1">(</span><span class="mtk12 bracket-highlighting-1">)</span><span class="mtk1">&nbsp;</span><span class="mtk18">for</span><span class="mtk1">&nbsp;sample&nbsp;</span><span class="mtk17">in</span><span class="mtk1">&nbsp;bootstrap_samples</span><span class="mtk12 bracket-highlighting-0">]</span></span></div><div style="top:342px;height:19px;" class="view-line"><span><span class="mtk1">bootstrap_percentile_blood_pressure&nbsp;=&nbsp;</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk1">np.percentile</span><span class="mtk12 bracket-highlighting-1">(</span><span class="mtk1">sample</span><span class="mtk12 bracket-highlighting-2">[</span><span class="mtk5">'BloodPressure'</span><span class="mtk12 bracket-highlighting-2">]</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">95</span><span class="mtk12 bracket-highlighting-1">)</span><span class="mtk1">&nbsp;</span><span class="mtk18">for</span><span class="mtk1">&nbsp;sample&nbsp;</span><span class="mtk17">in</span><span class="mtk1">&nbsp;bootstrap_samples</span><span class="mtk12 bracket-highlighting-0">]</span></span></div><div style="top:361px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:380px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Calculate&nbsp;population&nbsp;statistics</span></span></div><div style="top:399px;height:19px;" class="view-line"><span><span class="mtk1">pop_mean_blood_pressure&nbsp;=&nbsp;data</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'BloodPressure'</span><span class="mtk12 bracket-highlighting-0">]</span><span class="mtk1">.mean</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:418px;height:19px;" class="view-line"><span><span class="mtk1">pop_std_blood_pressure&nbsp;=&nbsp;data</span><span class="mtk12 bracket-highlighting-0">[</span><span class="mtk5">'BloodPressure'</span><span class="mtk12 bracket-highlighting-0">]</span><span class="mtk1">.std</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:437px;height:19px;" class="view-line"><span><span class="mtk1">pop_percentile_blood_pressure&nbsp;=&nbsp;np.percentile</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">data</span><span class="mtk12 bracket-highlighting-1">[</span><span class="mtk5">'BloodPressure'</span><span class="mtk12 bracket-highlighting-1">]</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">95</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:456px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:475px;height:19px;" class="view-line"><span><span class="mtk8">#&nbsp;Create&nbsp;boxplots&nbsp;for&nbsp;comparison</span></span></div><div style="top:494px;height:19px;" class="view-line"><span><span class="mtk1">plt.figure</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">figsize=</span><span class="mtk12 bracket-highlighting-1">(</span><span class="mtk6">10</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">6</span><span class="mtk12 bracket-highlighting-1">)</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:513px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:532px;height:19px;" class="view-line"><span><span class="mtk1">plt.subplot</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk6">3</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">1</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">1</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:551px;height:19px;" class="view-line"><span><span class="mtk1">plt.boxplot</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">bootstrap_mean_blood_pressure</span><span class="mtk12">,</span><span class="mtk1">&nbsp;vert=</span><span class="mtk9">False</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:570px;height:19px;" class="view-line"><span><span class="mtk1">plt.axvline</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">x=pop_mean_blood_pressure</span><span class="mtk12">,</span><span class="mtk1">&nbsp;color=</span><span class="mtk5">'red'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;linestyle=</span><span class="mtk5">'--'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;label=</span><span class="mtk5">'Population&nbsp;Mean'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:589px;height:19px;" class="view-line"><span><span class="mtk1">plt.xlabel</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Mean&nbsp;Blood&nbsp;Pressure'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:608px;height:19px;" class="view-line"><span><span class="mtk1">plt.title</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Bootstrap&nbsp;Mean&nbsp;Blood&nbsp;Pressure&nbsp;vs&nbsp;Population&nbsp;Mean'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:627px;height:19px;" class="view-line"><span><span class="mtk1">plt.legend</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:646px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:665px;height:19px;" class="view-line"><span><span class="mtk1">plt.subplot</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk6">3</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">1</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">2</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:684px;height:19px;" class="view-line"><span><span class="mtk1">plt.boxplot</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">bootstrap_std_blood_pressure</span><span class="mtk12">,</span><span class="mtk1">&nbsp;vert=</span><span class="mtk9">False</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:703px;height:19px;" class="view-line"><span><span class="mtk1">plt.axvline</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">x=pop_std_blood_pressure</span><span class="mtk12">,</span><span class="mtk1">&nbsp;color=</span><span class="mtk5">'red'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;linestyle=</span><span class="mtk5">'--'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;label=</span><span class="mtk5">'Population&nbsp;Std'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:722px;height:19px;" class="view-line"><span><span class="mtk1">plt.xlabel</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Standard&nbsp;Deviation&nbsp;of&nbsp;Blood&nbsp;Pressure'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:741px;height:19px;" class="view-line"><span><span class="mtk1">plt.title</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Bootstrap&nbsp;Std&nbsp;Blood&nbsp;Pressure&nbsp;vs&nbsp;Population&nbsp;Std'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:760px;height:19px;" class="view-line"><span><span class="mtk1">plt.legend</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:779px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:798px;height:19px;" class="view-line"><span><span class="mtk1">plt.subplot</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk6">3</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">1</span><span class="mtk12">,</span><span class="mtk1">&nbsp;</span><span class="mtk6">3</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:817px;height:19px;" class="view-line"><span><span class="mtk1">plt.boxplot</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">bootstrap_percentile_blood_pressure</span><span class="mtk12">,</span><span class="mtk1">&nbsp;vert=</span><span class="mtk9">False</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:836px;height:19px;" class="view-line"><span><span class="mtk1">plt.axvline</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk1">x=pop_percentile_blood_pressure</span><span class="mtk12">,</span><span class="mtk1">&nbsp;color=</span><span class="mtk5">'red'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;linestyle=</span><span class="mtk5">'--'</span><span class="mtk12">,</span><span class="mtk1">&nbsp;label=</span><span class="mtk5">'Population&nbsp;95th&nbsp;Percentile'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:855px;height:19px;" class="view-line"><span><span class="mtk1">plt.xlabel</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'95th&nbsp;Percentile&nbsp;of&nbsp;Blood&nbsp;Pressure'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:874px;height:19px;" class="view-line"><span><span class="mtk1">plt.title</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk5">'Bootstrap&nbsp;95th&nbsp;Percentile&nbsp;Blood&nbsp;Pressure&nbsp;vs&nbsp;Popul</span><span class="mtk5">ation&nbsp;95th&nbsp;Percentile'</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:893px;height:19px;" class="view-line"><span><span class="mtk1">plt.legend</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:912px;height:19px;" class="view-line"><span><span></span></span></div><div style="top:931px;height:19px;" class="view-line"><span><span class="mtk1">plt.tight_layout</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:950px;height:19px;" class="view-line"><span><span class="mtk1">plt.show</span><span class="mtk12 bracket-highlighting-0">(</span><span class="mtk12 bracket-highlighting-0">)</span></span></div><div style="top:969px;height:19px;" class="view-line"><span><span></span></span></div></div><div data-mprt="1" class="contentWidgets" style="position: absolute; top: 0px;"></div><div role="presentation" aria-hidden="true" class="cursors-layer cursor-line-style cursor-solid"><div class="cursor monaco-mouse-cursor-text " style="height: 19px; top: 0px; left: 0px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; display: block; visibility: hidden; padding-left: 0px; width: 1.66667px;"></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute; width: 1154px; height: 10px; left: 0px; bottom: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; width: 1154px;"></div></div><canvas class="decorationsOverviewRuler" aria-hidden="true" width="16" height="1197" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; top: 0px; right: 0px; width: 14px; height: 998px; will-change: unset; display: block;"></canvas><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute; width: 14px; height: 998px; right: 0px; top: 0px;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 14px; transform: translate3d(0px, 0px, 0px); contain: strict; will-change: unset; height: 998px;"></div></div></div><div role="presentation" aria-hidden="true" style="width: 1174px;"></div><textarea data-mprt="6" class="inputarea monaco-mouse-cursor-text" wrap="on" autocorrect="off" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Editor content;Press Alt+F1 for Accessibility Options." tabindex="0" role="textbox" aria-roledescription="editor" aria-multiline="true" aria-haspopup="false" aria-autocomplete="both" style="tab-size: 15.3984px; font-family: monospace, Consolas, &quot;Courier New&quot;, monospace; font-weight: normal; font-size: 14px; font-feature-settings: &quot;liga&quot; 0, &quot;calt&quot; 0; font-variation-settings: normal; line-height: 19px; letter-spacing: 0px; top: 0px; left: 6px; width: 1174px; height: 1px;"></textarea><div class="monaco-editor-background textAreaCover" style="position: absolute; top: 0px; left: 0px; width: 0px; height: 0px;"></div><div data-mprt="4" class="overlayWidgets" style="width: 1174px;"></div><div data-mprt="8" class="minimap slider-mouseover" role="presentation" aria-hidden="true" style="position: absolute; left: 0px; width: 0px; height: 998px;"><div class="minimap-shadow-hidden" style="height: 998px;"></div><canvas width="0" height="1197" style="position: absolute; left: 0px; width: 0px; height: 998px;"></canvas><canvas class="minimap-decorations-layer" width="0" height="1197" style="position: absolute; left: 0px; width: 0px; height: 998px;"></canvas><div class="minimap-slider" style="position: absolute; transform: translate3d(0px, 0px, 0px); contain: strict; width: 0px; will-change: unset;"><div class="minimap-slider-horizontal" style="position: absolute; width: 0px; height: 0px;"></div></div></div><div role="presentation" aria-hidden="true" class="blockDecorations-container"></div></div><div data-mprt="2" class="overflowingContentWidgets" style="display: none;"><div widgetid="editor.contrib.resizableContentHoverWidget" style="position: fixed; height: 10px; width: 10px; z-index: 50; display: none; visibility: hidden; max-width: 1600px; top: 265.141px; left: 201.484px;"><div class="monaco-sash vertical" style="left: 8px;"></div><div class="monaco-sash vertical disabled" style="left: -2px;"></div><div class="monaco-sash orthogonal-edge-north horizontal" style="top: -2px;"><div class="orthogonal-drag-handle end"></div></div><div class="monaco-sash orthogonal-edge-south horizontal disabled" style="top: 8px;"><div class="orthogonal-drag-handle end"></div></div><div class="monaco-hover hidden" role="tooltip" style="width: 0px; height: 0px;" tabindex="0"><div class="monaco-scrollable-element " role="presentation" style="position: relative; overflow: hidden;"><div class="monaco-hover-content" style="overflow: hidden; font-size: 14px; line-height: 1.35714; max-width: 774.84px; max-height: 250px; width: 0px; height: 0px;"><div class="hover-row markdown-hover"><div class="hover-contents"><div class="rendered-markdown"><p><code>float64: pop_percentile_blood_pressure</code></p></div></div></div><div class="hover-row markdown-hover"><div class="hover-contents"><div class="rendered-markdown"><p><code>90.0</code></p></div></div></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar horizontal" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; height: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div role="presentation" aria-hidden="true" class="invisible scrollbar vertical" style="position: absolute;"><div class="slider" style="position: absolute; top: 0px; left: 0px; width: 10px; transform: translate3d(0px, 0px, 0px); contain: strict;"></div></div><div class="shadow"></div><div class="shadow"></div><div class="shadow"></div></div></div></div></div><div class=".in-cell-overflowing"><div widgetid="editor.contrib.quickInputWidget" style="position: absolute; top: 0px; right: 50%;"></div></div></div></div></div><colab-form class="formview vertical layout flex"><div class="widget-area vertical layout"></div></colab-form></div>
    <div class="output"><!----> <div class="output-header"> </div>
        <div class="output-content">
          <div class="output-info"><colab-output-info title="Clear output

executed at 11:11 PM (0 minutes ago)
executed in 1.065s" hovering=""><template shadowrootmode="open"><!----> <md-icon class="collaborator account-circle" alt="Execution output" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->account_circle</md-icon>
      <!--?lit$2926288814$--> <mwc-icon-button icon="cancel" command="clear-focused-or-selected-outputs" alt="Clear output"><template shadowrootmode="open"><!----><button class="mdc-icon-button mdc-icon-button--display-flex" aria-label="cancel"><!--?lit$2926288814$-->
    <!--?lit$2926288814$--><i class="material-icons"><!--?lit$2926288814$-->cancel</i>
    <span><slot></slot></span>
  </button></template>
          </mwc-icon-button><!--?--></template></colab-output-info></div>
          <div class="output-iframe-container">
            <div class="output-iframe-sizer" style="min-height: 0px;"> <div><div class="outputview" style="height: 607px;"><colab-dom-lifecycle-events style="display: none;"></colab-dom-lifecycle-events><iframe allow="accelerometer; autoplay; camera; gyroscope; magnetometer; microphone; serial; usb; xr-spatial-tracking; clipboard-write" sandbox="allow-downloads allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-storage-access-by-user-activation allow-modals allow-popups-to-escape-sandbox" src="./untitled10_files/outputframe(2).html" class="" style="height: 607px;"></iframe></div></div><div><div></div></div></div>
          </div>
        </div></div></div><colab-cell-next-steps><template shadowrootmode="open"><!----></template></colab-cell-next-steps></div></div><div class="add-cell">
      <div class="add-cell-buttons">
        <md-outlined-button class="add-code add-button" aria-label="Add code cell
Ctrl+M B" title="Add code cell
Ctrl+M B" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add code cell
Ctrl+M B">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Code
        </md-outlined-button>
        <md-outlined-button class="add-text add-button" aria-label="Add text cell" title="Add text cell" role="presentation" value="" has-icon=""><template shadowrootmode="open" shadowrootdelegatesfocus><!---->
      <!--?lit$2926288814$--><div class="outline"></div>
      <div class="background"></div>
      <md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
      <md-ripple for="button" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
      <!--?lit$2926288814$--><button id="button" class="button" aria-label="Add text cell">
      <!--?lit$2926288814$-->
      <span class="touch"></span>
      <!--?lit$2926288814$--><slot name="icon"></slot>
      <span class="label"><slot></slot></span>
      <!--?lit$2926288814$-->
    
    </button>
    </template>
          <md-icon slot="icon" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>add</md-icon>
          <!--?lit$2926288814$-->Text
        </md-outlined-button>
        <!--?lit$2926288814$-->
      </div><hr>
    </div></div></div>
              </div>
            </div>
          </div>
          <!--?lit$2926288814$--> <div class="footer-links">
      <a target="_blank" href="https://colab.research.google.com/signup?utm_source=footer&amp;utm_medium=link&amp;utm_campaign=footer_links">
        <!--?lit$2926288814$-->Colab paid products
      </a>
      -
      <a href="https://colab.research.google.com/cancel-subscription" target="_blank">
        <!--?lit$2926288814$-->Cancel contracts here
      </a>
    </div>
        </div>
      </colab-shaded-scroller>
      <div class="notebook-scroll-shadow" style=""></div>
    </div></colab-tab></div>
  </div></colab-tab-pane>
      <colab-resizer style="height: 33.3%" class="sn-resize no-tabs"><div class="resizer-thumb"></div>
        <!--?lit$2926288814$--><colab-tab-pane class="layout vertical grow no-tabs" align="horizontal"><!----> <div class="layout vertical grow">
    <div class="tab-pane-header layout horizontal noshrink">
      <md-tabs><template shadowrootmode="open"><!---->
      <div class="tabs">
        <slot></slot>
      </div>
      <md-divider part="divider"><template shadowrootmode="open"><!----></template></md-divider>
    </template></md-tabs>
      <div class="layout grow"></div>
      <!--?lit$2926288814$--> <md-icon-button title="Show more" aria-label="Show more" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Show more">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
    <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_horiz</md-icon>
  </md-icon-button>
    </div>
    <div class="layout vertical grow tab-pane-container"> </div>
  </div></colab-tab-pane>
      </colab-resizer>
    </div>
      <colab-resizer style="width: 37%" class="we-resize no-tabs"><div class="resizer-thumb"></div>
        <!--?lit$2926288814$--> <div class="layout vertical tab-pane-parent">
      <!--?lit$2926288814$--><colab-tab-pane class="layout vertical grow no-tabs" align="horizontal"><!----> <div class="layout vertical grow">
    <div class="tab-pane-header layout horizontal noshrink">
      <md-tabs><template shadowrootmode="open"><!---->
      <div class="tabs">
        <slot></slot>
      </div>
      <md-divider part="divider"><template shadowrootmode="open"><!----></template></md-divider>
    </template></md-tabs>
      <div class="layout grow"></div>
      <!--?lit$2926288814$--> <md-icon-button title="Show more" aria-label="Show more" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Show more">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
    <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_horiz</md-icon>
  </md-icon-button>
    </div>
    <div class="layout vertical grow tab-pane-container"> </div>
  </div></colab-tab-pane>
      <colab-resizer style="height: 33.3%" class="sn-resize no-tabs"><div class="resizer-thumb"></div>
        <!--?lit$2926288814$--><colab-tab-pane class="layout vertical grow no-tabs" align="horizontal"><!----> <div class="layout vertical grow">
    <div class="tab-pane-header layout horizontal noshrink">
      <md-tabs><template shadowrootmode="open"><!---->
      <div class="tabs">
        <slot></slot>
      </div>
      <md-divider part="divider"><template shadowrootmode="open"><!----></template></md-divider>
    </template></md-tabs>
      <div class="layout grow"></div>
      <!--?lit$2926288814$--> <md-icon-button title="Show more" aria-label="Show more" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Show more">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
    <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>more_horiz</md-icon>
  </md-icon-button>
    </div>
    <div class="layout vertical grow tab-pane-container"> </div>
  </div></colab-tab-pane>
      </colab-resizer>
    </div>
      </colab-resizer>
    </div></colab-tab-layout-container>
        </div>
        <div class="proxies"><div><colab-dom-lifecycle-events style="display: none;"></colab-dom-lifecycle-events><iframe allow="" sandbox="allow-downloads allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-storage-access-by-user-activation allow-popups-to-escape-sandbox" src="./untitled10_files/outputframe(3).html" style="width: 1px; height: 1px; position: absolute; top: -100px;"></iframe></div><div><colab-dom-lifecycle-events style="display: none;"></colab-dom-lifecycle-events><iframe allow="accelerometer; autoplay; camera; gyroscope; magnetometer; microphone; serial; usb; xr-spatial-tracking; clipboard-write" sandbox="allow-downloads allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-storage-access-by-user-activation allow-modals allow-popups-to-escape-sandbox" src="./untitled10_files/outputframe(4).html" style="width: 1px; height: 1px; position: absolute; top: -100px;"></iframe></div></div>
      <colab-file-viewer-manager></colab-file-viewer-manager></div>
    <colab-status-bar role="region" aria-label="Runtime status bar" style="min-height: inherit;"><template shadowrootmode="open"><!----> <!--?lit$2926288814$--> <div class="connect-status">
        <md-icon status="icon-okay" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$--><svg viewBox="0 0 24 24"><!--?lit$2926288814$--><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></svg></md-icon>
        <div aria-atomic="true" aria-live="polite"><!--?lit$2926288814$-->Connected to Python 3 Google Compute Engine backend</div>
      </div>
      <md-icon-button class="visible-on-closed" title="Connected" aria-label="Connected" disabled="" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Connected" disabled="">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple disabled="" aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
        <md-icon filled="" class="visible-on-closed" status="icon-okay" aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template><!--?lit$2926288814$-->fiber_manual_record</md-icon>
      </md-icon-button>
      <md-icon-button title="Close" aria-label="Close" role="presentation" value=""><template shadowrootmode="open" shadowrootdelegatesfocus><!----><button id="button" class="icon-button  standard " aria-label="Close">
        <!--?lit$2926288814$--><md-focus-ring part="focus-ring" for="button" aria-hidden="true"><template shadowrootmode="open"><!----></template></md-focus-ring>
        <!--?lit$2926288814$--><md-ripple aria-hidden="true"><template shadowrootmode="open"><!----><div class="surface   "></div></template></md-ripple>
        <!--?lit$2926288814$--><span class="icon"><slot></slot></span>
        <!--?lit$2926288814$-->
        <!--?lit$2926288814$--><span class="touch"></span>
        <!--?lit$2926288814$-->
  </button></template>
        <md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>close</md-icon>
      </md-icon-button></template></colab-status-bar></div><div class="goog-menu" id="file-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="new" class="goog-menuitem " role="menuitem" id=":2" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->New notebook<!--?lit$2926288814$--></div></div><div command="open" class="goog-menuitem " role="menuitem" id=":3" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Open notebook<!--?lit$2926288814$--></div></div><div command="import-notebook" class="goog-menuitem " role="menuitem" id=":4" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Upload notebook<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":5" style="user-select: none;"></div><div command="rename" class="goog-menuitem " role="menuitem" id=":6" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Rename<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":7" style="user-select: none;"></div><div command="clone" class="goog-menuitem " role="menuitem" id=":8" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Save a copy in Drive<!--?lit$2926288814$--></div></div><div command="copy-to-gist" class="goog-menuitem " role="menuitem" id=":9" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Save a copy as a GitHub Gist<!--?lit$2926288814$--></div></div><div command="copy-to-github" class="goog-menuitem " role="menuitem" id=":a" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Save a copy in GitHub<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":b" style="user-select: none;"></div><div command="save" class="goog-menuitem " role="menuitem" id=":c" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Save<!--?lit$2926288814$--></div></div><div command="show-history" class="goog-menuitem " role="menuitem" id=":d" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Revision history<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":e" style="user-select: none;"></div><div class="goog-submenu goog-menuitem" id="download-submenu-menu-button" role="menuitem" aria-haspopup="true" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;">
      <!--?lit$2926288814$-->Download
    <span class="goog-submenu-arrow" style="user-select: none;">►</span></div></div><div command="print" class="goog-menuitem " role="menuitem" id=":i" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Print<!--?lit$2926288814$--></div></div></div><div class="goog-menu" id="download-submenu-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="download-ipynb" class="goog-menuitem " role="menuitem" id=":g" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Download .ipynb<!--?lit$2926288814$--></div></div><div command="download-python" class="goog-menuitem " role="menuitem" id=":h" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Download .py<!--?lit$2926288814$--></div></div></div><div class="goog-menu" id="edit-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="undo" class="goog-menuitem " role="menuitem" id=":k" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Undo<!--?lit$2926288814$--></div></div><div command="redo" class="goog-menuitem " role="menuitem" id=":l" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Redo<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":m" style="user-select: none;"></div><div command="select-all" class="goog-menuitem " role="menuitem" id=":n" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Select all cells<!--?lit$2926288814$--></div></div><div command="cut" class="goog-menuitem " role="menuitem" id=":o" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Cut cell or selection<!--?lit$2926288814$--></div></div><div command="copy" class="goog-menuitem " role="menuitem" id=":p" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Copy cell or selection<!--?lit$2926288814$--></div></div><div command="paste" class="goog-menuitem " role="menuitem" id=":q" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Paste<!--?lit$2926288814$--></div></div><div command="delete-cell-or-selection" class="goog-menuitem " role="menuitem" id=":r" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Delete selected cells<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":s" style="user-select: none;"></div><div command="find" class="goog-menuitem " role="menuitem" id=":t" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Find and replace<!--?lit$2926288814$--></div></div><div command="find-next" class="goog-menuitem " role="menuitem" id=":u" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Find next<!--?lit$2926288814$--></div></div><div command="find-previous" class="goog-menuitem " role="menuitem" id=":v" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Find previous<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":w" style="user-select: none;"></div><div command="notebook-settings" class="goog-menuitem " role="menuitem" id=":x" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Notebook settings<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":y" style="user-select: none;"></div><div command="clear-outputs" class="goog-menuitem " role="menuitem" id=":z" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Clear all outputs<!--?lit$2926288814$--></div></div></div><div class="goog-menu" id="view-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="show-toc-pane" class="goog-menuitem goog-option" role="menuitemcheckbox" aria-checked="false" id=":11" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><div class="goog-menuitem-checkbox" style="user-select: none;"><!----><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>check</md-icon> </div><!--?lit$2926288814$-->Table of contents<!--?lit$2926288814$--></div></div><div command="show-fileinfo" class="goog-menuitem " role="menuitem" id=":12" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Notebook info<!--?lit$2926288814$--></div></div><div command="show-executedcode" class="goog-menuitem " role="menuitem" id=":13" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Executed code history<!--?lit$2926288814$--></div></div><div command="toggle-comments-visibility" class="goog-menuitem goog-option" role="menuitemcheckbox" aria-checked="false" id=":14" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><div class="goog-menuitem-checkbox" style="user-select: none;"><!----><md-icon aria-hidden="true"><template shadowrootmode="open"><!----><slot></slot></template>check</md-icon> </div><!--?lit$2926288814$-->Comments sidebar<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":15" style="user-select: none;"></div><div command="collapse-sections" class="goog-menuitem " role="menuitem" id=":16" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Collapse sections<!--?lit$2926288814$--></div></div><div command="expand-sections" class="goog-menuitem " role="menuitem" id=":17" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Expand sections<!--?lit$2926288814$--></div></div><div command="save-section-layout" class="goog-menuitem " role="menuitem" id=":18" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Save collapsed section layout<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":19" style="user-select: none;"></div><div command="hide-code" class="goog-menuitem " role="menuitem" id=":1a" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Show/hide code<!--?lit$2926288814$--></div></div><div command="toggle-output" class="goog-menuitem " role="menuitem" id=":1b" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Show/hide output<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":1c" style="user-select: none;"></div><div command="focus-next-tab" class="goog-menuitem " role="menuitem" id=":1d" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Focus next tab<!--?lit$2926288814$--></div></div><div command="focus-previous-tab" class="goog-menuitem " role="menuitem" id=":1e" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Focus previous tab<!--?lit$2926288814$--></div></div><div command="move-tab-to-next" class="goog-menuitem " role="menuitem" id=":1f" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Move tab to next pane<!--?lit$2926288814$--></div></div><div command="move-tab-to-prev" class="goog-menuitem " role="menuitem" id=":1g" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Move tab to previous pane<!--?lit$2926288814$--></div></div></div><div class="goog-menu" id="insert-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="insert-cell-below" class="goog-menuitem " role="menuitem" id=":1i" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Code cell<!--?lit$2926288814$--></div></div><div command="add-text" class="goog-menuitem " role="menuitem" id=":1j" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Text cell<!--?lit$2926288814$--></div></div><div command="add-section-header" class="goog-menuitem " role="menuitem" id=":1k" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Section header cell<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":1l" style="user-select: none;"></div><div command="open-scratch-code-cell" class="goog-menuitem " role="menuitem" id=":1m" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Scratch code cell<!--?lit$2926288814$--></div></div><div command="snippets" class="goog-menuitem " role="menuitem" id=":1n" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Code snippets<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":1o" style="user-select: none;"></div><div command="add-form-field" class="goog-menuitem " role="menuitem" id=":1p" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Add a form field<!--?lit$2926288814$--></div></div></div><div class="goog-menu" id="runtime-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="runall" class="goog-menuitem " role="menuitem" id=":1r" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Run all<!--?lit$2926288814$--></div></div><div command="runbefore" class="goog-menuitem " role="menuitem" id=":1s" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Run before<!--?lit$2926288814$--></div></div><div command="runfocused" class="goog-menuitem " role="menuitem" id=":1t" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Run the focused cell<!--?lit$2926288814$--></div></div><div command="runselected" class="goog-menuitem " role="menuitem" id=":1u" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Run selection<!--?lit$2926288814$--></div></div><div command="runafter" class="goog-menuitem " role="menuitem" id=":1v" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Run after<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":1w" style="user-select: none;"></div><div command="interrupt" class="goog-menuitem " role="menuitem" id=":1x" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Interrupt execution<!--?lit$2926288814$--></div></div><div command="restart" class="goog-menuitem " role="menuitem" id=":1y" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Restart session<!--?lit$2926288814$--></div></div><div command="restart-and-run-all" class="goog-menuitem " role="menuitem" id=":1z" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Restart session and run all<!--?lit$2926288814$--></div></div><div command="powerwash-current-vm" class="goog-menuitem " role="menuitem" id=":20" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Disconnect and delete runtime<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":21" style="user-select: none;"></div><div command="change-runtime-type" class="goog-menuitem " role="menuitem" id=":22" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Change runtime type<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":23" style="user-select: none;"></div><div command="manage-sessions" class="goog-menuitem " role="menuitem" id=":24" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Manage sessions<!--?lit$2926288814$--></div></div><div command="open-resource-viewer" class="goog-menuitem " role="menuitem" id=":25" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->View resources<!--?lit$2926288814$--></div></div><div command="view-runtime-logs" class="goog-menuitem " role="menuitem" id=":26" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->View runtime logs<!--?lit$2926288814$--></div></div></div><div class="goog-menu" id="tools-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="show-command-palette" class="goog-menuitem " role="menuitem" id=":28" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Command palette<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":29" style="user-select: none;"></div><div command="preferences" class="goog-menuitem " role="menuitem" id=":2a" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Settings<!--?lit$2926288814$--></div></div><div command="shortcuts" class="goog-menuitem " role="menuitem" id=":2b" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Keyboard shortcuts<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":2c" style="user-select: none;"></div><div command="open-differ" class="goog-menuitem " role="menuitem" id=":2d" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Diff notebooks<!--?lit$2926288814$--> <span class="screenreader-only" style="user-select: none;"><!--?lit$2926288814$-->(opens in a new tab)</span></div></div></div><div class="goog-menu" id="help-menu" role="menu" aria-haspopup="true" style="display: none; user-select: none;"><!--?lit$2926288814$--><div command="faq" class="goog-menuitem " role="menuitem" id=":2f" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Frequently asked questions<!--?lit$2926288814$--></div></div><div command="view-relnotes" class="goog-menuitem " role="menuitem" id=":2g" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->View release notes<!--?lit$2926288814$--></div></div><div command="snippets" class="goog-menuitem " role="menuitem" id=":2h" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Search code snippets<!--?lit$2926288814$--></div></div><div class="goog-menuseparator goog-menuitem-disabled" aria-disabled="true" role="separator" id=":2i" style="user-select: none;"></div><div command="report-bug" class="goog-menuitem " role="menuitem" id=":2j" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Report a bug<!--?lit$2926288814$--></div></div><div command="send-feedback" class="goog-menuitem " role="menuitem" id=":2k" style="user-select: none;"><div class="goog-menuitem-content" style="user-select: none;"><!--?lit$2926288814$-->Send feedback<!--?lit$2926288814$--></div></div></div><div class="thumbnail"></div><div class="monaco-aria-container"><div class="monaco-alert" role="alert" aria-atomic="true"></div><div class="monaco-alert" role="alert" aria-atomic="true"></div><div class="monaco-status" role="complementary" aria-live="polite" aria-atomic="true"></div><div class="monaco-status" role="complementary" aria-live="polite" aria-atomic="true"></div></div><iframe id="apiproxya8324bb615760b02d9f7e78784afe1ebf2d14bc20.1450290365" name="apiproxya8324bb615760b02d9f7e78784afe1ebf2d14bc20.1450290365" src="./untitled10_files/proxy.html" tabindex="-1" aria-hidden="true" style="width: 1px; height: 1px; position: absolute; top: -100px; display: none;"></iframe><iframe src="./untitled10_files/bscframe.html" style="display: none;"></iframe><div><div class="grecaptcha-badge" data-style="bottomright" style="width: 256px; height: 60px; position: fixed; visibility: hidden; display: block; transition: right 0.3s ease 0s; bottom: 14px; right: -186px; box-shadow: gray 0px 0px 5px; border-radius: 2px; overflow: hidden;"><div class="grecaptcha-logo"><iframe title="reCAPTCHA" width="256" height="60" role="presentation" name="a-w9xkhhbzxrt7" frameborder="0" scrolling="no" sandbox="allow-forms allow-popups allow-same-origin allow-scripts allow-top-navigation allow-modals allow-popups-to-escape-sandbox allow-storage-access-by-user-activation" src="./untitled10_files/anchor.html"></iframe></div><div class="grecaptcha-error"></div><textarea id="g-recaptcha-response-100000" name="g-recaptcha-response" class="g-recaptcha-response" style="width: 250px; height: 40px; border: 1px solid rgb(193, 193, 193); margin: 10px 25px; padding: 0px; resize: none; display: none;"></textarea></div><iframe style="display: none;" src="./untitled10_files/saved_resource.html"></iframe></div><div id="hl-aria-live-message-container" aria-live="polite" class="visually-hidden"></div><div id="hl-aria-live-alert-container" role="alert" aria-live="assertive" class="visually-hidden"></div><iframe id="hfcr" src="./untitled10_files/RotateCookiesPage.html" style="display: none;"></iframe></body></html>