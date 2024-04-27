using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.Random;
using static PathTracer.Samplers;

namespace PathTracer
{
    class PathTracer
    {
        // <summary>
        // Given Ray r and Scene s, trace the ray over the scene and return the estimated radiance
        // </summary>
        // <param name="r">Ray direction</param>
        // <param name="s">Scene to trace</param>
        // <returns>Estimated radiance in the ray direction</returns>
        public Spectrum Li(Ray r, Scene s)
        {
            var L = Spectrum.ZeroSpectrum;
            Spectrum beta = Spectrum.Create(1);
            int bounces = 0;

            while (bounces < 20)
            {
                (double? closest_t, SurfaceInteraction surf_info) = s.Intersect(r);

                // No intersection
                if (!closest_t.HasValue || surf_info == null) { break; }


                // Hit light
                Spectrum emittedSpectrum = surf_info.Le(-r.d); //klicemo Le funkcijo, ce ni luc vrne 0, direction of the ray je -r.d
                if (surf_info.Obj is Light)
                {
                    Debug.WriteLine("We hit a light source!");
                    if (bounces == 0) L = beta * emittedSpectrum;
                    //L = beta * emittedSpectrum;
                    break;
                }
                else // Didnt hit light
                {

                    // Sample the light 
                    Spectrum sampledLight = Light.UniformSampleOneLight(surf_info, s);
                    L.AddTo(beta * sampledLight);

                    // Sample a new direction using BSDF, f -> intensity of color contribution in direction wiW, pdf -> p(wi)
                    Primitive primitive = surf_info.Obj;
                    Shape shape = primitive as Shape;
                    if (shape == null) { break; }


                    (Spectrum f, Vector3 wi, double pdf, bool isSpecular) = shape.BSDF.Sample_f(-r.d, surf_info);

                    // Create a new ray from point surf_info in direction wi
                    r = surf_info.SpawnRay(wi);

                    // Update beta 
                    beta = beta * f * Vector3.AbsDot(wi, surf_info.Normal) / pdf;

                    // Russian roulette, small beta -> break, larger beta -> continue
                    if (bounces > 3)
                    {
                        double q = 1 - beta.Max();
                        if (ThreadSafeRandom.NextDouble() < q) { break; }
                        beta = beta / (1 - q);
                    }


                    bounces++;

                }

            }// end while

            return L;
        } // end Spectrum Li

    }
}
