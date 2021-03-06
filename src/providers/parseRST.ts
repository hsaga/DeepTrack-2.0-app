import restructured from 'restructured';

// TODO: Convert to TSX with react elements

const rst2html = (rstSource, indent = 2) => {
    const parsedRST = restructured.parse(rstSource);
    console.log(parsedRST);
    return render_any(parsedRST, 0, 2);
};

interface rstItem {
    type: string;
    depth?: number;
    value?: string;
    bullet?: string;
    role?: string;
    children: rstItem[];
}

const render_any = (element: rstItem, level = 0, indent = 2) => {
    switch (element.type) {
        case 'document':
            return render_document(element, level, indent);
        case 'section':
            return render_section(element, level, indent);
        case 'transition':
            return render_transition(element, level, indent);
        case 'paragraph':
            return render_paragraph(element, level, indent);
        case 'bullet_list':
            return render_bullet_list(element, level, indent);
        case 'enumerated_list':
            return render_enumerated_list(element, level, indent);
        case 'definition_list':
            return render_definition_list(element, level, indent);
        case 'list_item':
            return render_list_item(element, level, indent);
        case 'line':
            return render_line(element, level, indent);
        case 'line_block':
            return render_line_block(element, level, indent);
        case 'literal_block':
            return render_literal_block(element, level, indent);
        case 'block_quote':
            return render_block_quote(element, level, indent);
        case 'interpreted_text':
            return render_interpreted_text(element, level, indent);
        case 'text':
            return render_text(element, level, indent);
        case 'emphasis':
            return render_emphasis(element, level, indent);
        case 'strong':
            return render_strong(element, level, indent);
        case 'literal':
            return render_literal(element, level, indent);
        case 'directive':
            return render_code_block(element, level, indent);
        default:
            return render_unknown(element, level, indent);
    }
};

const render_unknown = (element: rstItem, level = 0, indent = 2) => {
    if (element.children) {
        return render_block_element('div', `rst-unknown rst-${element.type}`, element, level, indent);
    } else {
        return render_leaf_element('div', `rst-unknown rst-${element.type}`, element, level, indent);
    }
};

const render_document = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('div', 'rst-document', element, level, indent);
};

const render_section = (element: rstItem, level = 0, indent = 2) => {
    const indentString = ' '.repeat(level * indent);

    const title = render_title(element.depth || 0, element.children[0], level + 1, indent);

    const children = element.children
        .slice(1)
        .map((e) => render_any(e, level + 1, indent))
        .join('\n');

    return `${indentString}<div class="rst-section">\n${title}${children}${indentString}</div>\n`;
};

const render_title = (depth: number, element: rstItem, level = 0, indent = 2) => {
    const titleTag = `h${depth}`;
    const titleClassName = `rst-title-${depth}`;

    return render_block_element(titleTag, titleClassName, element, level, 0);
};

const render_transition = (element: rstItem, level = 0, indent = 2) => {
    // TODO: implement transitions
    return render_unknown(element, level, 0);
};

const render_paragraph = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('p', 'rst-paragraph', element, level, 0);
};

const render_bullet_list = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('ul', 'rst-bullet-list', element, level, 0);
};

const render_enumerated_list = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('ol', 'rst-enumerated-list', element, level, 0);
};

const render_definition_list = (element: rstItem, level = 0, indent = 2) => {
    // TODO: implement definition lists
    return render_unknown(element, level, 0);
};

const render_list_item = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('li', 'rst-list-item', element, level, 0);
};

const render_line = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('div', 'rst-line', element, level, 0);
};

const render_line_block = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('div', 'rst-line-block', element, level, 0);
};

const render_literal_block = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('pre', 'rst-literal-block', element, level, 0);
};

const render_block_quote = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('div', 'rst-block-quote', element, level, 0);
};

const render_text = (element: rstItem, level = 0, indent = 2) => {
    return render_leaf_element('span', 'rst-text', element, level, 0);
};

const render_interpreted_text = (element: rstItem, level = 0, indent = 2) => {
    const className = 'rst-interpreted_text' + (element.role ? ` rst-role-${element.role}` : '');
    return render_inline_element('span', className, element, level, 0);
};

const render_emphasis = (element: rstItem, level = 0, indent = 2) => {
    return render_inline_element('em', 'rst-emphasis', element, level, 0);
};

const render_strong = (element: rstItem, level = 0, indent = 2) => {
    return render_inline_element('strong', 'rst-strong', element, level, 0);
};

const render_literal = (element: rstItem, level = 0, indent = 2) => {
    return render_inline_element('tt', 'rst-literal', element, level, 0);
};

const render_code_block = (element: rstItem, level = 0, indent = 2) => {
    return render_block_element('div', 'rst-code_block', element, level, 1, '<br>');
};

const render_leaf_element = (
    tag: string | undefined,
    className: string | undefined,
    element: rstItem,
    level = 0,
    indent = 2,
) => {
    const indentString = ' '.repeat(level * indent);
    return `<${tag} class="${className}">${(element.value || '').replace(/\n$/, ' ')}</${tag}>`;
};

const render_inline_element = (
    tag: string | undefined,
    className: string | undefined,
    element: rstItem,
    level = 0,
    indent = 2,
) => {
    const children = element.children.map((e) => render_any(e, level + 1, indent)).join('');
    return `<${tag} class="${className}">${children}</${tag}>`;
};

const render_block_element = (
    tag: string | undefined,
    className: string | undefined,
    element: rstItem,
    level = 0,
    indent = 2,
    separator = '',
) => {
    const indentString = ' '.repeat(level * indent);
    const children = element.children.map((e) => render_any(e, level + 1, indent)).join(separator);

    return `${indentString}<${tag} class="${className}">\n${children}\n${indentString}</${tag}>\n`;
};

export default rst2html;
