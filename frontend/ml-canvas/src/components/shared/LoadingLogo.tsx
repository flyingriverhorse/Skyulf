import { useEffect, useState } from 'react';
import logoUrl from '../../../../../static/img/app_logo/ml_learning_swirl_yellow_v2.svg';
import animationUrl from '../../../../../static/img/app_logo/Swirl Logo Animation.mp4';

/** Animate the application mark during loading unless reduced motion is requested. */
export function LoadingLogo() {
  const [animate, setAnimate] = useState(false);
  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setAnimate(!preference.matches);
    update();
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  return animate ? (
    <video
      src={animationUrl}
      poster={logoUrl}
      autoPlay
      muted
      loop
      playsInline
      width={64}
      height={64}
      className="h-16 w-16 rounded-xl object-cover"
      aria-hidden="true"
    />
  ) : (
    <img src={logoUrl} alt="" width={64} height={64} className="h-16 w-16 object-contain" aria-hidden="true" />
  );
}
