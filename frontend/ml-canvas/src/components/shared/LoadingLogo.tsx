import { useEffect, useState } from 'react';
import logoUrl from '../../../../../static/img/app_logo/logo.png';
import { LoadingLogoAnimation } from './LoadingLogoAnimation';

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
    <LoadingLogoAnimation logoUrl={logoUrl} />
  ) : (
    <img src={logoUrl} alt="" width={64} height={64} className="h-16 w-16 object-contain" aria-hidden="true" />
  );
}
