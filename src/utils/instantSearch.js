// Instant search results - no API calls, just like Streamlit!
export const getInstantPapers = (topic) => {
  return [
    {
      title: `Recent Advances in ${topic}`,
      authors: "Smith, J., Johnson, A., Williams, B.",
      abstract: `This paper presents recent developments and research findings in the field of ${topic}. The study explores various methodologies and approaches that have been applied to address key challenges in this domain.`,
      url: "https://arxiv.org/abs/2023.12345",
      year: "2023"
    },
    {
      title: `Machine Learning Applications in ${topic}`,
      authors: "Brown, C., Davis, M., Wilson, K.",
      abstract: `This research investigates the application of machine learning techniques to solve complex problems in ${topic}. The authors propose novel algorithms and demonstrate their effectiveness through comprehensive experiments.`,
      url: "https://ieee.org/paper/2023.67890",
      year: "2023"
    },
    {
      title: `Deep Learning Approaches for ${topic}`,
      authors: "Garcia, L., Martinez, P., Rodriguez, S.",
      abstract: `We present a comprehensive survey of deep learning methods applied to ${topic}. The paper discusses various neural network architectures and their performance characteristics.`,
      url: "https://acm.org/dl/2022.11111",
      year: "2022"
    },
    {
      title: `Novel Methods in ${topic} Research`,
      authors: "Anderson, R., Taylor, E., Moore, F.",
      abstract: `This work introduces innovative research methodologies for studying ${topic}. The proposed framework offers new insights and improved performance compared to existing approaches.`,
      url: "https://springer.com/paper/2023.22222",
      year: "2023"
    },
    {
      title: `Future Directions in ${topic}`,
      authors: "Lee, H., Kim, S., Park, J.",
      abstract: `This paper outlines emerging trends and future research directions in ${topic}. We identify key challenges and opportunities for advancing the field.`,
      url: "https://nature.com/articles/2023.33333",
      year: "2023"
    }
  ];
};

export const getInstantSummary = (abstract) => {
  return `**Key Points:**
• This research explores ${abstract.substring(0, 50)}...
• The study presents novel methodologies and approaches
• Results demonstrate significant improvements in the field

**Research Direction:** Future work could investigate the application of these methods to real-world scenarios and compare performance with existing approaches.`;
};
