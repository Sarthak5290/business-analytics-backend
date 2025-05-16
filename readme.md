## Mermaid Diagram

```mermaid
graph TD
    A[User Onboarding & Industry Detection] -->|Provides Input For|> B(Step 2: Idea Submission & Expansion)
    B -->|Requires Details From|> C{User Input}
    C -->|If Provided|> D[Extracted Problem, Solution, and Target Audience]
    C -->|If Not Provided|> E[Cannot Proceed]
    D -->|Used For|> F[Market Opportunity Analysis]
    D -->|Used For|> G[Investor-Friendly Format]
    D -->|Used For|> H[Detailed Business Concept]
    F -->|Informs|> I[Industry, Target Market Cap, Expected Revenue]
    G -->|Informs|> I
    H -->|Informs|> I
    I -->|Determines|> J[Potential Business Models]
    I -->|Determines|> K[Startup Funding Options]
    J -->|Influences|> L[Personalized Onboarding Response]
    K -->|Influences|> L
    style A fill:#bbf, stroke:#f66, stroke-width:2px
    style B fill:#bbf, stroke:#f66, stroke-width:2px
    style C fill:#ccf, stroke:#f66, stroke-width:2px
    style D fill:#ddf, stroke:#f66, stroke-width:2px
    style E fill:#eed, stroke:#f66, stroke-width:2px
    style F fill:#bbf, stroke:#f66, stroke-width:2px
    style G fill:#bbf, stroke:#f66, stroke-width:2px
    style H fill:#bbf, stroke:#f66, stroke-width:2px
    style I fill:#ccf, stroke:#f66, stroke-width:2px
    style J fill:#ddf, stroke:#f66, stroke-width:2px
    style K fill:#ddf, stroke:#f66, stroke-width:2px
    style L fill:#eed, stroke:#f66, stroke-width:2px
```
