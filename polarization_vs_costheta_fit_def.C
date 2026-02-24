#ifdef __CLING__
#pragma cling optimize(0)
#endif
void polarization_vs_costheta_fit_def()
{
//=========Macro generated from canvas: canvas2/
//=========  (Fri Jan 16 12:36:58 2026) by ROOT version 6.32.04
   TCanvas *canvas2 = new TCanvas("canvas2", "",0,0,800,600);
   gStyle->SetOptStat(0);
   canvas2->SetHighLightColor(2);
   canvas2->Range(-1.5,-0.3920286,1.5,0.09996368);
   canvas2->SetFillColor(0);
   canvas2->SetBorderMode(0);
   canvas2->SetBorderSize(2);
   canvas2->SetFrameBorderMode(0);
   canvas2->SetFrameBorderMode(0);
   
   Double_t Graph_fx1001[20] = { -0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
   0.75, 0.85, 0.95 };
   Double_t Graph_fy1001[20] = { -0.0003612218, -0.002033949, -0.006088148, -0.01334101, -0.0239462, -0.03816745, -0.0571691, -0.0801993, -0.1057443, -0.1349221, -0.1639796, -0.1920353, -0.2172733, -0.2393731, -0.2570825, -0.2705093, -0.2803157,
   -0.2869399, -0.2906661, -0.292262 };
   Double_t Graph_fex1001[20] = { 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
   0.05, 0.05, 0.05 };
   Double_t Graph_fey1001[20] = { 0.01832619, 0.01566072, 0.0170399, 0.01816138, 0.01717804, 0.01744078, 0.01846143, 0.01826722, 0.0192744, 0.01892624, 0.01925676, 0.01863601, 0.01917143, 0.01817414, 0.01780124, 0.01760213, 0.01782293,
   0.01766954, 0.01571499, 0.01776787 };
   TGraphErrors *gre = new TGraphErrors(20,Graph_fx1001,Graph_fy1001,Graph_fex1001,Graph_fey1001);
   gre->SetName("");
   gre->SetTitle("");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   
   TH1F *Graph_Graph1001 = new TH1F("Graph_Graph1001","",100,-1.2,1.2);
   Graph_Graph1001->SetMinimum(-0.3428293);
   Graph_Graph1001->SetMaximum(0.05076445);
   Graph_Graph1001->SetDirectory(nullptr);
   Graph_Graph1001->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   Graph_Graph1001->SetLineColor(ci);
   Graph_Graph1001->GetXaxis()->SetTitle("cos #theta_{#tau}");
   Graph_Graph1001->GetXaxis()->SetLabelFont(42);
   Graph_Graph1001->GetXaxis()->SetTitleOffset(1);
   Graph_Graph1001->GetXaxis()->SetTitleFont(42);
   Graph_Graph1001->GetYaxis()->SetTitle("P_{#tau}(cos #theta_{#tau})");
   Graph_Graph1001->GetYaxis()->SetLabelFont(42);
   Graph_Graph1001->GetYaxis()->SetTitleOffset(1.4);
   Graph_Graph1001->GetYaxis()->SetTitleFont(42);
   Graph_Graph1001->GetZaxis()->SetLabelFont(42);
   Graph_Graph1001->GetZaxis()->SetTitleOffset(1);
   Graph_Graph1001->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_Graph1001);
   
   gre->Draw("ap");
   
   TF1 *fit_func1 = new TF1("fit_func","-([0]*(1+x*x)+2*[1]*x) / (1+x*x + 2*[1]*[0]*x)",-1,1, TF1::EAddToList::kDefault);
   fit_func1->SetFillColor(19);
   fit_func1->SetFillStyle(0);

   ci = TColor::GetColor("#ff0000");
   fit_func1->SetLineColor(ci);
   fit_func1->SetLineWidth(2);
   fit_func1->GetXaxis()->SetLabelFont(42);
   fit_func1->GetXaxis()->SetTitleOffset(1);
   fit_func1->GetXaxis()->SetTitleFont(42);
   fit_func1->GetYaxis()->SetLabelFont(42);
   fit_func1->GetYaxis()->SetTitleFont(42);
   fit_func1->SetParameter(0,0.1494877);
   fit_func1->SetParError(0,0);
   fit_func1->SetParLimits(0,0,0);
   fit_func1->SetParameter(1,0.1494331);
   fit_func1->SetParError(1,0);
   fit_func1->SetParLimits(1,0,0);
   fit_func1->Draw("SAME");
   
   TLegend *leg = new TLegend(0.5,0.65,0.85,0.89,NULL,"brNDC");
   leg->SetBorderSize(1);
   leg->SetLineColor(0);
   leg->SetLineStyle(1);
   leg->SetLineWidth(0);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("NULL","#sqrt{s}=91 GeV, 5.20  fb^{-1}","");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("","Pseudo Data","pl");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("fit_func","Fit for A_{#tau} and A_{e}","l");

   ci = TColor::GetColor("#ff0000");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(2);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   canvas2->Modified();
   canvas2->SetSelected(canvas2);
}
