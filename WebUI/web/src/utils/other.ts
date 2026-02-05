import { NTooltip } from 'naive-ui';
import { h } from 'vue';
import SvgIcon from '@/components/custom/svg-icon.vue';

export function renderTooltip(trigger: any, content: any) {
  return h(NTooltip, null, {
    trigger: () => trigger,
    default: () => content
  });
}

export function renderTooltip2(title: any, content: any) {
  return renderTooltip(
    h('div', { class: 'flex-center' }, [h('span', { class: 'mr-1' }, title), h(SvgIcon, { icon: 'mdi-help-circle' })]),
    content
  );
}
